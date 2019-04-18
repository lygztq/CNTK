//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNetwork.h"
#include "SimpleEvaluator.h"
#include "DataReader.h"
#include "ScriptableObjects.h"
#include "Criterion.h"
#include <vector>
#include <string>
#include <stdexcept>
#include "fileutil.h"
#include "Config.h"
#include <chrono>
#include <random>
#include "Profiler.h"
#include "MASGD.h"
#include "ASGDHelper.h"
#include <map>
#include "IDistGradAggregator.h"
using namespace std; // ugh! TODO: get rid of this from .h files!!!

#define CNTK_CHECKPOINT_VERSION_1 1     // 1 -> no version number 
#define CNTK_CHECKPOINT_VERSION_2 2      
#define CURRENT_CNTK_CHECKPOINT_VERSION CNTK_CHECKPOINT_VERSION_2

namespace CNTK { namespace Internal {
    // Forward declarations.
    class TensorBoardFileWriter;
    typedef std::shared_ptr<TensorBoardFileWriter> TensorBoardFileWriterPtr;
}}

namespace Microsoft { namespace MSR { namespace CNTK {

struct BestEpoch;

enum class LearningRateSearchAlgorithm : int
{
    None,
    AdjustAfterEpoch,
    SearchBeforeEpoch
};

enum class AdaptationRegType : int
{
    None,
    KL
};

enum class GradientsUpdateType : int
{
    None,
    AdaGrad,
    RmsProp,
    FSAdaGrad
};

// modelParallelSGD can be combined with dataParallelSGD/modelAveragingSGD/blockMomentumSGD 
// but dataParallelSGD/modelAveragingSGD/blockMomentumSGD are mutually exclusive (at least at the moment)
// we assign the lower 8 bits to the enumerate data parallelization methods 
// and next 8 bits to model parallelization methods
enum class ParallelizationMethod : int
{
    none = 0,
    dataParallelSGD = 1,
    modelAveragingSGD = 2,
    blockMomentumSGD = 3,
    dataParallelASGD = 4,
    modelParallelSGD = (1 << 8) // Currently unsupported
};


enum class AdjustType : int
{
    None,
    Poly,
    Inv,
    Exp,
    Step
};


// configuration parameters associated with RMSProp learning algorithm
struct RMSPropInfo
{
    double gamma;
    double inc;
    double dec;
    double max;
    double min;

    RMSPropInfo()
    {
        gamma = 0.99;
        inc = 1.2;
        dec = 0.75;
        max = 10.0;
        min = 0.1;
    }
};

struct GradientUpdateInfo
{
    GradientsUpdateType type = GradientsUpdateType::AdaGrad;
    float gaussianNoiseInjectStd = 0.0075f;

    // for FSAdaGrad:
    double targetAdagradAvDenom = 1;
    size_t varianceTimeConstant = 2 * 3600 * 100; // originally was: 2h of speech
};


// learning rate adjust per iteration info
struct LRAPIInfo
{
    AdjustType adjustType;
    size_t iter = 0;
    size_t maxIter;
    size_t step;
    double base_;
    double gamma;
    double power;
    size_t numItersToShowLR;
    size_t numItersToSaveModel;
    bool reachMaxIter = false;
    size_t sgdTraceLevel;

    LRAPIInfo() {}
};


struct BestEpoch
{
    double criterionMinValue = numeric_limits<double>::max();
    int32_t epochIndex = -1;
};

// ---------------------------------------------------------------------------
// SGDParams -- parameters for SGD
//
// TODO: This should keep everything that is configured by the config.
//       Currently it does not store which matrices are used.
// ---------------------------------------------------------------------------

struct SGDParams : public ScriptableObjects::Object
{
    template <class ConfigRecord> // (needed for default value of m_gradientBits)
    SGDParams(const ConfigRecord& configSGD, size_t sizeofElemType);

    SGDParams(const ScriptableObjects::IConfigRecordPtr configp);

    // SGDParams(SGDParams&&) = default; // (does not compile in VS 2013; not critical)

    size_t GetMaxEpochs() { return m_maxEpochs; }

protected:
    // learning rate per sample provided outside
    floatargvector m_learningRatesParam;
    intargvector m_learningRatesSpecifiedForMBSize; // 1 for per sample, m_mbSize[] for per MB
    floatargvector m_momentumParam;
    intargvector m_momentumSpecifiedForMBSize;
    bool m_useNesterovMomentum;
    bool m_disableMomentumUnitGain;

    // Determine the MB size used for mapping a given learning-rate or momentum parameter to a per-sample value.
    // MB size is the number of samples across all time steps and parallel sequences.
    // This function exists to post-fix a design bug in SGD:
    // In the case of BPTT, the 'minibatchSize' parameter given to the SGD module really means the truncation size,
    // while the MB size to be used is (truncation size * number of parallel sequences).
    // SGD also does not know #parallel sequences upfront.
    size_t FixUpEffectiveMBSize(size_t specifiedMBSize, size_t numParallelSequences) const
    {
        // remedy the bug that truncation size is incorrectly passed as MB size
        if (m_truncated && specifiedMBSize > 1)      // currently only happens in this mode
        {
            if (numParallelSequences == 0)
            {
                RuntimeError("Learning rate and momentum are not supported per minibatch, please specify them per sample.");
            }

            specifiedMBSize *= numParallelSequences; // assume 'specifiedMBSize' refers to truncation size
        }
        // end bug post-fix
        // TODO: This ^^ should go away once SGD gets fixed to take the truncation size as a parameter.

        return specifiedMBSize;
    }

    // helpers to convert learning rates to per-sample values used in the actual algorithms
    // 'numParallelSequences' must be specified because of the definitional MB-size bug in SGD mentioned above, and should go away once that is sorted out.
    double GetLearningRatePerSample(size_t epoch /*BUGBUG workaround:*/, size_t numParallelSequences) const
    {
        return m_learningRatesParam[epoch] / FixUpEffectiveMBSize(m_learningRatesSpecifiedForMBSize[epoch], numParallelSequences);
    }
    double GetMomentumPerSample(size_t epoch /*BUGBUG workaround:*/, size_t numParallelSequences) const
    {
        return pow(m_momentumParam[epoch], 1.0 / FixUpEffectiveMBSize(m_momentumSpecifiedForMBSize[epoch], numParallelSequences));
    }

    ParallelizationMethod GetParallelizationMethod() const
    {
        if (m_mpi == nullptr)
            return ParallelizationMethod::none;

        return m_parallelizationMethod;
    }

    // helper function to initialize and check BlockMomentumSGD related parameters
    void InitializeAndCheckBlockMomentumSGDParameters();
    // only true when the user specify LearningRatePerMB and the number of parallel utterances in Reader > 1
    // bool m_needToNormalizeLRByParallUtterance;          // TODO: should go away
    // bool m_needToNormalizeMomentumByParallUtterance;

    intargvector m_mbSize;
    bool m_truncated; // do BPTT
    // BUGBUG: The 'Truncated' option is duplicated in the reader and must be set to the same there (e.g. by defining in the config on an outer enclosing level, like current samples).
    //         We really should only read it in SGD and pass it ourselves on to the Reader, instead of it being a Reader parameter.
    // BUGBUG: If m_truncated, then m_mbSize is interpreted as truncation length; the actual MB size is a combination of that and the #parallel sequences specified in the reader.
    // TODO: do not specify 'Truncated' but 'TruncatedLength', set m_truncated so given, and let m_mbSize control how many #parallel sequences the reader is allowed to pack into an MB.
    size_t m_maxSamplesInRAM;
    // This is related with subminibatch implementation
    // maxSamplesInRAM denotes how many samples we used in forward-backward on net.
    // Due to the GPU memory limitations, it is sometime not possible to hold the m_mbSize in RAM.
    // To mitigate this issue, we adopt the sub-minibatch implementation, where
    // each m_mbSize[epoch] is divided by a few sub-minibatch of which size will be no more than m_maxSamplesInRAM
    // a forward-backward is performed for each sub-minibatch; a model update is performed after each minibatch
    size_t m_numSubminiBatches;
    // alternative method to specify how to split minibatches into subminibatches
    // default is 1, which means no subminibatch is used
    // if m_maxTempMemSizeInSamples = SIZE_MAX (which means users do not specify the option) and m_numSubminiBatches > 1
    // we divide one minibatch to m_numSubminiBatches subMinibatches

    // the number of samples in each epoch (0 means, use all the samples in each epoch).
    size_t m_epochSize;
    size_t m_maxComputedEpochSize;

    // the total number of epochs to run.
    size_t m_maxEpochs;

    bool m_gradientClippingWithTruncation;
    double m_clippingThresholdPerSample;

    intargvector m_numSamples4Search;
    size_t m_numBestSearchEpoch;

    // Threshold size in bytes for single gradient to do packing
    size_t m_packThresholdSizeInBytes;

    LearningRateSearchAlgorithm m_autoLearnRateSearchType;

    AdaptationRegType m_adaptationRegType;
    double m_adaptationRegWeight;
    bool m_needAdaptRegularization;

    bool m_loadBestModel;
    double m_reduceLearnRateIfImproveLessThan;
    bool m_continueReduce;

    // determine after how many epochs the learning rate should be auto adjusted.
    size_t m_learnRateAdjustInterval;

    bool m_useCVSetControlLRIfCVExists;
    bool m_useEvalCriterionControlLR;

    double m_increaseLearnRateIfImproveMoreThan;
    double m_learnRateIncreaseFactor;
    double m_learnRateDecreaseFactor;
    bool m_autoAdjustMinibatch;
    size_t m_minibatchSearchCriterionErrorMargin;
    size_t m_minibatchSizeTuningFrequency;
    size_t m_minibatchSizeTuningMax;

    doubleargvector m_dropoutRates;
    doubleargvector m_batchNormalizationTimeConstant;
    doubleargvector m_batchNormalizationBlendTimeConstant;
    size_t m_maxTempMemSizeInSamplesForCNN;

    int m_traceLevel;

    size_t m_numPrevLearnRates;

    double m_minLearnRate;

    GradientUpdateInfo m_gradType;
    RMSPropInfo m_rpi;

    size_t m_numMBsToShowResult = 0;
    size_t m_firstMBsToShowResult = 0;
    int m_numMBsToCUDAProfile;

    std::wstring m_tensorBoardLogDir;
    size_t m_tensorBoardNumMBsToLogResult;

    bool m_doGradientCheck;
    double m_gradientCheckSigDigit;

    bool m_doUnitTest;

    bool m_useAllDataForPreComputedNode;

    // Parallel training
    MPIWrapperPtr m_mpi;

    ParallelizationMethod m_parallelizationMethod;
    bool m_enableDistributedMBReading;
    // indicates if we're using default value of the m_enableDistributedMBReading flag
    // (in which case, it can potentially be overriden).
    // This flag is only relevant for the new (V2) readers. It exist because of
    // a shortcoming in DecimateMinibatchInPlace, which does not yet work when inputs 
    // in the same minibatch have different layouts, which is something only V2 readers can
    // produce. 
    bool m_enableDistributedMBReadingNotSpecified; 
    int m_parallelizationStartEpochNum;

    // decide if/how often we measure and show sync performance stats (seconds spend on sync, seconds since last sync etc.) ?
    // 0: No sync perfomance stats
    // 1: Show stats on every sync
    // n > 1: Show stats after every n sync
    int m_syncStatsTrace;

    // Data parallel SGD training parameters
    intargvector m_numGradientBits;
    bool m_bufferedAsyncGradientAggregation;
    bool m_zeroThresholdFor1Bit;

    // Parallel training related with MA / BM
    size_t m_modelAggregationBlockSize;
    bool   m_resetSGDMomentum; 
    bool   m_useNesterovBlockMomentum;
    double m_blockLearningRate; 
    double m_blockMomentumAsTimeConstant;

    bool m_needAveMultiplier;
    double m_L2RegWeight;
    double m_L1RegWeight;

    // Parallel training related with ASGD 
    intargvector m_nSyncSamplesPerWorker;
    bool m_isAsyncBufferEnabled;
    bool m_isSimulateMA;
    AdjustLearningRateAtBeginning m_adjustLearningRateAtBeginning;
    double m_adjustCoefficient;
    size_t m_adjustPerMinibatches;

    // sequence training
    double m_hSmoothingWeight;
    double m_frameDropThresh;
    bool m_doReferenceAlign;
    double m_seqGammarCalcAMF;
    double m_seqGammarCalcLMF;
    double m_seqGammarCalcWP;
    double m_seqGammarCalcbMMIFactor;
    bool m_seqGammarCalcUsesMBR;

    // decide whether should apply regularization into BatchNormalizationNode
    // true: disable Regularization
    // false: enable Regularization (default)
    bool m_disableRegInBatchNormalization;


    LRAPIInfo m_lrapiInfo;

	// mixed precision training parameters
	float m_mixedTrainLossScaleFactor;
};

template <class ElemType>
class IDistGradAggregator;
class IMixTypedDistGradAggregator;
struct GradientPackage;

// -----------------------------------------------------------------------
// class SGD
// -----------------------------------------------------------------------

// TODO: make this independent of ElemType. Then these repeated dynamic_pointer_casts will go away
// TODO: why is this a class, and not just a procedure? Then we wouldn't have to include the massive header
template <class ElemType>
class SGD : public SGDParams
{
protected:
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    typedef ClassBasedCrossEntropyWithSoftmaxNode<ElemType>* ClassBasedCrossEntropyWithSoftmaxNodePtr;

public:
    // constructor from old CNTK config. This is a function template that is also used to get the config from Scripting.
    template <class ConfigRecordType>
    SGD(const ConfigRecordType& configSGD)
        : SGDParams(configSGD, sizeof(ElemType)),
          // TODO: The next few do not belong into SGD any more than the network or reader we operate on. Either move network and reader in here, or move these out.
          m_modelPath((const wstring&) configSGD(L"modelPath")),
          m_keepCheckPointFiles(configSGD(L"keepCheckPointFiles", false)),
          m_saveBestModelPerCriterion(configSGD(L"saveBestModelPerCriterion", false)),
          m_trainCriterionNodeName((const wstring&) configSGD(L"trainCriterionNodeName", L"")),
          m_evalCriterionNodeName ((const wstring&) configSGD(L"evalCriterionNodeName", L"")),
          m_traceNodeNamesReal    (configSGD(L"traceNodeNamesReal",     ConfigRecordType::Array(stringargvector()))),
          m_traceNodeNamesCategory(configSGD(L"traceNodeNamesCategory", ConfigRecordType::Array(stringargvector()))),
          m_traceNodeNamesSparse  (configSGD(L"traceNodeNamesSparse",   ConfigRecordType::Array(stringargvector()))),
          m_prevChosenMinibatchSize(0),
          m_lastFinishedEpochTrainLoss(0.0),
          m_distGradAgg(nullptr),
          m_gradHeader(nullptr)
    {
        msra::files::make_intermediate_dirs(m_modelPath);
    }
    // note: This must be in the header, as we cannot properly specialize this constructor in the CPP to make sure all versions are generated.

    // constructor from Scripting
    SGD(const ScriptableObjects::IConfigRecordPtr configp)
        : SGD(*configp)
    {
    }

    void InitMPI(const MPIWrapperPtr& mpi)
    {
        m_mpi = mpi;

        if (m_mpi == nullptr)
            m_parallelizationMethod = ParallelizationMethod::none;
        }

    void Train(shared_ptr<ComputationNetwork> net, DEVICEID_TYPE deviceId,
               IDataReader* trainSetDataReader,
               IDataReader* validationSetDataReader, int startEpoch, bool loadNetworkFromCheckpoint);
    void Adapt(wstring origModelFileName, wstring refNodeName,
               IDataReader* trainSetDataReader,
               IDataReader* validationSetDataReader,
               const DEVICEID_TYPE deviceID, const bool makeMode = true);

	// mixed precision training
	bool UseMixedPrecisionTraining() { return std::is_same<ElemType, half>::value; }

protected:

    const std::vector<ComputationNodeBasePtr>& GetTrainCriterionNodes(ComputationNetworkPtr net);
    const std::vector<ComputationNodeBasePtr>& GetEvalCriterionNodes(ComputationNetworkPtr net);

    void TrainOrAdaptModel(int startEpoch, ComputationNetworkPtr net,
                           bool networkLoadedFromCheckpoint,
                           ComputationNetworkPtr refNet,
                           ComputationNodeBasePtr refNode,
                           IDataReader* trainSetDataReader,
                           IDataReader* validationSetDataReader);

protected:

    // return true if precomputation is executed.
    bool PreCompute(ComputationNetworkPtr net,
                    IDataReader* trainSetDataReader,
                    const std::vector<ComputationNodeBasePtr>& featureNodes,
                    const std::vector<ComputationNodeBasePtr>& labelNodes,
                    StreamMinibatchInputs* inputMatrices);

    // return a reasonable initial learning rate based on the initial mbsize
    double SearchForBestLearnRate(ComputationNetworkPtr net,
                                  ComputationNetworkPtr refNet,
                                  const ComputationNodeBasePtr& refNode, const int epochNumber,
                                  const double curLearnRate,
                                  IDataReader* trainSetDataReader,
                                  const std::vector<ComputationNodeBasePtr>& featureNodes,
                                  const std::vector<ComputationNodeBasePtr>& labelNodes,
                                  const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                  const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                                  StreamMinibatchInputs* inputMatrices,
                                  const std::list<ComputationNodeBasePtr>& learnableNodes,
                                  std::list<MatrixBasePtr>& smoothedGradients, std::vector<double> smoothedCounts,
                                  const bool learnRateInitialized,
                                  const double largestPrevLearnRatePerSample);

    void TrainOneMiniEpochAndReloadModel(ComputationNetworkPtr net,
                                         ComputationNetworkPtr refNet,
                                         const ComputationNodeBasePtr& refNode, const int epochNumber,
                                         const size_t epochSize, IDataReader* trainSetDataReader,
                                         const double learnRatePerSample,
                                         const size_t minibatchSize,
                                         const std::vector<ComputationNodeBasePtr>& featureNodes,
                                         const std::vector<ComputationNodeBasePtr>& labelNodes,
                                         const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                         const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                                         StreamMinibatchInputs* inputMatrices,
                                         const std::list<ComputationNodeBasePtr>& learnableNodes,
                                         std::list<MatrixBasePtr>& smoothedGradients, std::vector<double> smoothedCounts,
                                         /*out*/ EpochCriterion& epochCriterion,
                                         /*out*/ std::vector<EpochCriterion>& epochEvalErrors,
                                         std::string prefixMsg,
                                         const size_t maxNumOfSamples);

    size_t AdaptiveMinibatchSizing(ComputationNetworkPtr net,
                                   ComputationNetworkPtr refNet,
                                   const ComputationNodeBasePtr& refNode,
                                   const int epochNumber,
                                   const size_t numFramesToUseInSearch,
                                   IDataReader* trainSetDataReader,
                                   const double learnRatePerSample,
                                   const size_t initialMinibatchSize,
                                   const std::vector<ComputationNodeBasePtr>& featureNodes,
                                   const std::vector<ComputationNodeBasePtr>& labelNodes,
                                   const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                   const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                                   StreamMinibatchInputs* inputMatrices,
                                   const std::list<ComputationNodeBasePtr>& learnableNodes,
                                   std::list<MatrixBasePtr>& smoothedGradients, std::vector<double> smoothedCounts,
                                   const double learningRateAdjustmentFactor);

    // uses a small percentage of training data of minibatch to
    // speculatively train with various MB sizes; then picks the best
    size_t SearchForBestMinibatchSize(ComputationNetworkPtr net,
                                      ComputationNetworkPtr refNet,
                                      const ComputationNodeBasePtr& refNode,
                                      const int epochNumber,
                                      const size_t numFramesToUseInSearch,
                                      IDataReader* trainSetDataReader,
                                      const double learnRatePerSample,
                                      const std::vector<ComputationNodeBasePtr>& featureNodes,
                                      const std::vector<ComputationNodeBasePtr>& labelNodes,
                                      const std::vector<ComputationNodeBasePtr>& criterionNodes,
                                      const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                                      StreamMinibatchInputs* inputMatrices,
                                      const std::list<ComputationNodeBasePtr>& learnableNodes,
                                      std::list<MatrixBasePtr>& smoothedGradients, std::vector<double> smoothedCounts,
                                      const size_t minMinibatchSize, const size_t maxMinibatchSize);

    // Attempts to compute the error signal for the whole utterance, which will
    // be fed to the neural network as features. Currently it is a workaround
    // for the two-forward-pass sequence and ctc training, which allows
    // processing more utterances at the same time. Only used in Kaldi2Reader.
    // TODO: move the two-forward-pass support out of the reader.
    void AttemptUtteranceDerivativeFeatures(ComputationNetworkPtr net,
                                            IDataReader* trainSetDataReader,
                                            const std::vector<ComputationNodeBasePtr>& featureNodes,
                                            StreamMinibatchInputs* inputMatrices);

    size_t TrainOneEpoch(ComputationNetworkPtr net,
                         ComputationNetworkPtr refNet,
                         const ComputationNodeBasePtr& refNode,
                         const int epochNumber,
                         const size_t epochSize,
                         IDataReader* trainSetDataReader,
                         const double learnRatePerSample,
                         size_t tunedMBSize,
                         const std::vector<ComputationNodeBasePtr>& featureNodes,
                         const std::vector<ComputationNodeBasePtr>& labelNodes,
                         const std::vector<ComputationNodeBasePtr>& criterionNodes,
                         const std::vector<ComputationNodeBasePtr>& evaluationNodes,
                         StreamMinibatchInputs* inputMatrices,
                         const std::list<ComputationNodeBasePtr>& learnableNodes,
                         std::list<MatrixBasePtr>& smoothedGradients, std::vector<double>& smoothedCounts,
                         /*out*/ EpochCriterion& epochCriterion,
                         /*out*/ std::vector<EpochCriterion>& epochEvalErrors,
                         const std::string& prefixMsg = "",
                         const size_t maxNumberOfSamples = SIZE_MAX,
                         const size_t totalMBsSeenBefore = 0,
                         ::CNTK::Internal::TensorBoardFileWriterPtr tensorBoardWriter = nullptr,
                         const int startEpoch = 0);

    void InitDistGradAgg(int numEvalNodes, int numGradientBits, int deviceId, int traceLevel);
    void InitModelAggregationHandler(int traceLevel, DEVICEID_TYPE devID);
public:
    // UpdateWeights() - actual weight update, implementing various update rules
	template <class ActualElemType = ElemType>
    void UpdateWeightsImpl(Matrix<ActualElemType>& functionValues, Matrix<ActualElemType>& gradientValues,
						   Matrix<ActualElemType>& smoothedGradient, double& smoothedCount,
                       const double learnRatePerSample, const double momentumPerSample,
                       size_t actualMBSize,
                       const double L2RegWeight, const double L1RegWeight,
                       const bool needAveMultiplier,
                       const bool useNesterovMomentum,
                       const bool disableMomentumUnitGain) const;
	
	template <class NodeElemType>
	void UpdateWeights(std::shared_ptr<ComputationNode<NodeElemType>> learnableNode,
					   MatrixBasePtr smoothedGradient, double& smoothedCount,
					   shared_ptr<ComputationNetwork> net,
					   const double learnRatePerSample,
					   const int epochNumber,
					   size_t numSamplesInMinibatch);
	void MixedUpdateWeights(std::shared_ptr<ComputationNode<half>> learnableNode,
						    MatrixBasePtr smoothedGradient, double& smoothedCounts,
						    shared_ptr<ComputationNetwork> net,
						    const double learnRatePerSample,
						    const int epochNumber,
						    size_t numSamplesInMinibatch);

    // return -1 if nothing exists
    int DetermineStartEpoch(const bool makeMode);

    wstring GetModelNameForEpoch(const int epoch, bool bLastModel = false) const;

protected:

	template <class ActualElemType = ElemType>
    void ClipGradient(Matrix<ActualElemType>& gradient, const size_t actualMBSize) const;

    void SaveCheckPointInfo(const size_t epoch, const size_t totalSamplesSeen, // TODO: combine totalSamplesSeen and prevCriterion into a EpochCriterion type
                            const double learnRatePerSample,
                            const std::list<MatrixBasePtr>& smoothedGradients,
                            const std::vector<double>& smoothedCounts,
                            const double prevCriterion,
                            const size_t minibatchSize);

    bool TryLoadCheckPointInfo(const size_t epochNumber,
                               /*out*/ size_t& totalSamplesSeen,
                               /*out*/ double& learnRatePerSample,
                               std::list<MatrixBasePtr>& smoothedGradients,
                               std::vector<double>& smoothedCounts,
                               /*out*/ double& prevCriterion,
                               /*out*/ size_t& minibatchSize);
    void LoadCheckPointInfo(const size_t epochNumber,
                            /*out*/ size_t& totalSamplesSeen,
                            /*out*/ double& learnRatePerSample,
                            std::list<MatrixBasePtr>& smoothedGradients,
                            std::vector<double>& smoothedCounts,
                            /*out*/ double& prevCriterion,
                            /*out*/ size_t& minibatchSize);

    wstring GetCheckPointFileNameForEpoch(const int epoch);

    GradientsUpdateType GradUpdateType() const
    {
        return m_gradType.type;
    }
    double GradientUpdateNoiseStd() const
    {
        return m_gradType.gaussianNoiseInjectStd;
    }

public:
#define EPSILON 1e-5

    bool GradientCheck(ComputationNetworkPtr net,
                       const std::vector<ComputationNodeBasePtr>& criterionNodes,
                       const std::list<ComputationNodeBasePtr>& learnableNodes,
                       int npos);

protected:
    std::wstring m_modelPath;
    bool m_keepCheckPointFiles;
    bool m_saveBestModelPerCriterion;
    // Mapping from criterion to the best epoch on validation data set.
    std::map<std::wstring, BestEpoch> m_criteriaBestEpoch;

    std::wstring m_trainCriterionNodeName;
    std::wstring m_evalCriterionNodeName;

    // enable tracing. Nodes listed here get their m_traceNodeValueXXX flags set
    std::vector<std::wstring> m_traceNodeNamesReal;
    std::vector<std::wstring> m_traceNodeNamesCategory;
    std::vector<std::wstring> m_traceNodeNamesSparse;

    size_t m_prevChosenMinibatchSize;
    double m_lastFinishedEpochTrainLoss;

    std::shared_ptr<IDistGradAggregator<ElemType>> m_distGradAgg; // aggregate gradients
    std::shared_ptr<struct DistGradHeader> m_gradHeader; // aggregate criterion and errors

    shared_ptr<IMASGD<ElemType>> m_pMASGDHelper;

private:
    void MarkDropoutNodesEvalTimeStampAsOutdated(const ComputationNetworkPtr& net, const ComputationNodeBasePtr& criterionNode);
    std::shared_ptr<ASGDHelper<ElemType>> m_pASGDHelper;

    bool UsingGradientAggregation(size_t epochNumber) const
    {
        return ((GetParallelizationMethod() == ParallelizationMethod::dataParallelSGD) && (epochNumber >= m_parallelizationStartEpochNum));
    }

    bool UsingModelAggregation(size_t epochNumber) const
    {
        return ((GetParallelizationMethod() == ParallelizationMethod::modelAveragingSGD ||
                 GetParallelizationMethod() == ParallelizationMethod::blockMomentumSGD) &&
                (epochNumber >= m_parallelizationStartEpochNum));
    }

    bool UsingAsyncGradientAggregation(size_t epochNumber)
    {
        return ((GetParallelizationMethod() == ParallelizationMethod::dataParallelASGD) && (epochNumber >= m_parallelizationStartEpochNum));
    }

    bool UsingParallelTrain(size_t epochNumber)
    {
        return UsingGradientAggregation(epochNumber) || UsingModelAggregation(epochNumber) || UsingAsyncGradientAggregation(epochNumber);
    }

    void SynchronizeWorkers()
    {
        if (m_mpi != nullptr && GetParallelizationMethod() != ParallelizationMethod::dataParallelASGD)
        {
            m_mpi->WaitAll();
        }
        if (m_mpi != nullptr && GetParallelizationMethod() == ParallelizationMethod::dataParallelASGD)
        {
            m_pASGDHelper->WaitAll();
        }
        return;
    }
};

}}}
