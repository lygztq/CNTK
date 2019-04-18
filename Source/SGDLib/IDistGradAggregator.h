#pragma once

#include "DistGradHeader.h"
#include "MPIWrapper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// ---------------------------------------------------------------------------
// GradientPackage: Package used for gradient aggregation in parallel training
// ---------------------------------------------------------------------------
struct GradientPackage
{
	std::vector<std::shared_ptr<Matrix<float>>> m_gradFloat;
	std::vector<std::shared_ptr<Matrix<half>>> m_gradHalf;
	std::vector<std::shared_ptr<Matrix<double>>> m_gradDouble;

	bool IsEmpty() { return m_gradFloat.empty() && m_gradHalf.empty() && m_gradDouble.empty(); }
	int DeviceId() 
	{
		if (!m_gradFloat.empty()) return m_gradFloat[0]->GetDeviceId();
		else if (!m_gradHalf.empty()) return m_gradHalf[0]->GetDeviceId();
		else if (!m_gradDouble.empty()) return m_gradDouble[0]->GetDeviceId();
		else
			RuntimeError("The Gradient Package is empty.");
	}
};


template <class ElemType>
class IDistGradAggregator
{
public:
    IDistGradAggregator(const MPIWrapperPtr& mpi)
        : m_mpi(mpi)
    {}

    virtual ~IDistGradAggregator()
    {}

    // Returns a boolean indicating if any samples were processed
    virtual bool AggregateGradients(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool resetState) = 0;

    size_t NumProc()
    {
        return m_mpi->NumNodesInUse();
    }

    size_t MyRank()
    {
        return m_mpi->CurrentNodeRank();
    }

    void WaitAll()
    {
        m_mpi->WaitAll();
    }

protected:
    MPIWrapperPtr m_mpi;
};

class IMixTypedDistGradAggregator
{
public:
	IMixTypedDistGradAggregator(const MPIWrapperPtr& mpi)
		: m_mpi(mpi) 
	{}

	virtual ~IMixTypedDistGradAggregator() {}

	virtual bool AggregateGradients(const GradientPackage& gradients, DistGradHeader* headerCPU, bool resetState) = 0;

	size_t NumProc()
	{
		return m_mpi->NumNodesInUse();
	}

	size_t MyRank()
	{
		return m_mpi->CurrentNodeRank();
	}

	void WaitAll()
	{
		m_mpi->WaitAll();
	}

protected:
	MPIWrapperPtr m_mpi;
};

#define UsingIDistGradAggregatorMembers           \
    \
protected:                                        \
    using IDistGradAggregator<ElemType>::m_mpi;   \
    using IDistGradAggregator<ElemType>::NumProc; \
    using IDistGradAggregator<ElemType>::MyRank
} } }
