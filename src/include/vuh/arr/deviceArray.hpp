#pragma once

#include "arrayProperties.h"
#include "arrayIter.hpp"
#include "arrayUtils.h"
#include "allocDevice.hpp"
#include "basicArray.hpp"
#include "hostArray.hpp"

#include <vuh/traits.hpp>

#include <algorithm>
#include <cassert>

namespace vuh {
namespace arr {

/// Array class not supposed to take part in data exchange with host.
/// The only valid use for such arrays is passing them as (in or out) argument to a shader.
/// Resources allocation is handled by allocator defined by a template parameter which is supposed to provide
/// memory and anderlying vulkan buffer with suitable flags.
template<class T, class Alloc>
class DeviceOnlyArray: public BasicArray {
    using Base = BasicArray;
public:
	using value_type = T;
    DeviceOnlyArray() { }
   /// Constructs object of the class on given device.
   /// Memory is left unintitialized.
   DeviceOnlyArray( vuh::Device& device  ///< deice to create array on
	               , size_t n_elements    ///< number of elements
	               , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	               , vk::BufferUsageFlags flags_buffer={})   ///< additional (to defined by allocator) buffer usage flags
	   : BasicArray(device, n_elements*sizeof(T), flags_memory, flags_buffer, (Alloc *)nullptr)
	   , _size(n_elements)
	{}

   using Base::size_bytes;

private:
	uint32_t  _size; ///< number of elements
}; // class DeviceOnlyArray

/// Array with host data exchange interface suitable for memory allocated in device-local space.
/// Memory allocation and underlying buffer creation is managed by allocator defined by a template parameter.
/// Such allocator is expected to allocate memory in device local memory not mappable
/// for host access. However actual allocation may take place in a host-visible memory.
/// Some functions (like toHost(), fromHost()) switch to using the simplified data exchange methods
/// in that case. Some do not. In case all memory is host-visible (like on integrated GPUs) using this class
/// may result in performance penalty.
template<class T, class Alloc>
class DeviceArray: public BasicArray{
	using Base = BasicArray;
public:
	using value_type = T;
    DeviceArray() {}
	/// Create an instance of DeviceArray with given number of elements. Memory is uninitialized.
	DeviceArray( vuh::Device& device   ///< device to create array on
	           , size_t n_elements     ///< number of elements
	           , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	           , vk::BufferUsageFlags flags_buffer={})   ///< additional (to defined by allocator) buffer usage flags
	   : Base(device, n_elements*sizeof(T), flags_memory, flags_buffer, (Alloc *)nullptr)
	   , _size(n_elements)
	{}

	/// Create an instance of DeviceArray and initialize memory by content of some host iterable.
	template<class C, class=typename std::enable_if_t<vuh::traits::is_iterable<C>::value>>
	DeviceArray(vuh::Device& device  ///< device to create array on
	           , const C& c          ///< iterable to initialize from
	           , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	           , vk::BufferUsageFlags flags_buffer={})	  ///< additional (to defined by allocator) buffer usage flags
	   : DeviceArray(device, c.size(), flags_memory, flags_buffer)
	{
		using std::begin; using std::end;
		fromHost(begin(c), end(c));
	}

	/// Create an instance of DeviceArray and initialize it from a range of values.
	template<class It1, class It2>
   DeviceArray(vuh::Device& device   ///< device to create array on
	            , It1 begin           ///< range begin
	            , It2 end             ///< range end (points to one past the last element of the range)
	            , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	            , vk::BufferUsageFlags flags_buffer={})	///< additional (to defined by allocator) buffer usage flags
	   : DeviceArray(device, std::distance(begin, end), flags_memory, flags_buffer)
	{
		fromHost(begin, end);
	}

	/// Create an instance of DeviceArray of given size and initialize it using index based initializer function.
	template<class F>
	DeviceArray( vuh::Device& device  ///< device to create array on
	           , size_t n_elements    ///< number of elements
	           , F&& fun              ///< callable of a form function<T(size_t)> mapping an offset to array value
	           , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	           , vk::BufferUsageFlags flags_buffer={})	  ///< additional (to defined by allocator) buffer usage flags
	   : DeviceArray(device, n_elements, flags_memory, flags_buffer)
	{
		using std::begin;
		auto stage_buffer = HostArray<T, AllocDevice<properties::HostCoherent>>(Base::_dev, n_elements);
		auto stage_it = begin(stage_buffer);
		for(size_t i = 0; i < n_elements; ++i, ++stage_it){
			*stage_it = fun(i);
		}
		copyBuf(Base::_dev, stage_buffer, *this, size_bytes());
	}

    /// Destroy array, and release all associated resources.
    ~DeviceArray() noexcept {
       if(ptr) {
          Base::unmapMemory();
       }
    }

    /// Move constructor.
    DeviceArray(DeviceArray&& o): Base(std::move(o)), ptr(o.ptr), _size(o._size) {o.ptr = nullptr;}
     /// Move operator.
    auto operator=(DeviceArray&& o)-> DeviceArray& {
         this->swap(o);
         return *this;
    }
     /// doc me
     auto swap(DeviceArray& o) noexcept-> void {
         using std::swap;
         swap(static_cast<Base&>(*this), static_cast<Base&>(o));
       swap(ptr, o.ptr);
       swap(_size, o._size);
     }

   
	/// Copy data from host range to array memory.
	template<class It1, class It2>
	auto fromHost(It1 begin, It2 end)-> void {
		if(Base::isHostVisible()){
			std::copy(begin, end, host_data());
            Base::flush_mapped_writes();
            unmap_host_data();
		} else { // memory is not host visible, use staging buffer
			auto stage_buf = HostArray<T, AllocDevice<properties::HostCoherent>>(Base::_dev, begin, end);
			copyBuf(Base::_dev, stage_buf, *this, size_bytes());
		}
	}
    /// Copy data from host range to array memory.
    template<class It1, class It2, typename F>
    auto fromHost(It1 begin, It2 end, F&& fun)-> void {
        if(Base::isHostVisible()){
            std::transform(begin, end, host_data(), fun);
            Base::flush_mapped_writes();
            unmap_host_data();
        } else { // memory is not host visible, use staging buffer
            auto stage_buf = HostArray<T, AllocDevice<properties::HostCoherent>>(Base::_dev, begin, end, fun);
            copyBuf(Base::_dev, stage_buf, *this, size_bytes());
        }
    }
    /// Call fun to fill data to array memory.
    template<typename F>
    auto fromHost(F&& fun, size_t offset = 0, size_t size_ = 0)-> void {
        if (offset >= size())
            return;

        if(Base::isHostVisible()){
            fun(host_data() + offset);
            Base::flush_mapped_writes();
            unmap_host_data();
        } else { // memory is not host visible, use staging buffer
            auto stage_buf = HostArray<T, AllocDevice<properties::HostCoherent>>(Base::_dev, size_ ? size_ : size(), fun);
            auto offset_bytes = offset*sizeof(T);
            auto size = size_bytes() - offset_bytes;
            if (size_ != 0)
                size = std::min(size_*sizeof(T), stage_buf.size_bytes());
            copyBuf(Base::_dev, stage_buf, *this, size, 0u, offset_bytes);
        }
    }
   
	/// Copy data from host range to array memory with offset
	template<class It1, class It2>
	auto fromHost(It1 begin, It2 end, size_t offset)-> void {
        if (offset >= size())
            return;

		if(Base::isHostVisible()){
			std::copy(begin, end, host_data() + offset);
            Base::flush_mapped_writes();
            unmap_host_data();
		} else { // memory is not host visible, use staging buffer
			auto stage_buf = HostArray<T, AllocDevice<properties::HostCoherent>>(Base::_dev, begin, end);
            auto offset_bytes = offset*sizeof(T);
            auto size = size_bytes() - offset_bytes;
            size = std::min(size, stage_buf.size_bytes());
			copyBuf(Base::_dev, stage_buf, *this, size, 0u, offset_bytes);
		}
	}



   /// Copy data from array memory to host location indicated by iterator.
	/// The whole array data is copied over.
   template<class It, typename=typename std::enable_if_t<vuh::traits::is_host_iterator<It>::value>>
   auto toHost(It copy_to) const-> void {
      if(Base::isHostVisible()){
          auto copy_from = host_data();
          Base::invalidate_mapped_cache();
         std::copy_n(copy_from, size(), copy_to);
         unmap_host_data();
      } else {
         using std::begin; using std::end;
         auto stage_buf = HostArray<T, AllocDevice<properties::HostCached>>(Base::_dev, size());
         copyBuf(Base::_dev, *this, stage_buf, size_bytes());
         std::copy(begin(stage_buf), end(stage_buf), copy_to);
      }
   }
   
   /// Copy-transform data from array memory to host location indicated by iterator.
	/// The whole array data is transformed.
   template<class It, class F>
   auto toHost(It copy_to, F&& fun) const-> void {
      toHost(copy_to, size(), fun);
   }
   
   /// Copy-transform the chunk of data of given size from the beginning of array.
   template<class It, class F>
   auto toHost( It copy_to  ///< iterator indicating starting position to write to
	           , size_t size ///< number of elements to transform
	           , F&& fun     ///< transform function
	           ) const-> void
	{
		if(Base::isHostVisible()){
			auto copy_from = host_data();
            Base::invalidate_mapped_cache();
			std::transform(copy_from, copy_from + size, copy_to, std::forward<F>(fun));
            unmap_host_data();
		} else {
			using std::begin; using std::end;
			auto stage_buf = HostArray<T, AllocDevice<properties::HostCached>>(Base::_dev, size);
			copyBuf(Base::_dev, *this, stage_buf, size_bytes());
			std::transform(begin(stage_buf), end(stage_buf), copy_to, std::forward<F>(fun));
		}
	}
   /// Call back fun on data of array.
   template<class F, typename=typename std::enable_if_t<std::is_invocable_v<F, float*>> >
   auto toHost( F&& fun     ///< transform function
	           , size_t offset = 0) const-> void
	{
       if (offset >= size())
           return;

		if(Base::isHostVisible()){
			auto copy_from = host_data();
            Base::invalidate_mapped_cache();
            fun(copy_from + offset);
            unmap_host_data();
		} else {
			using std::begin; using std::end;
			auto stage_buf = HostArray<T, AllocDevice<properties::HostCached>>(Base::_dev, size());
			copyBuf(Base::_dev, *this, stage_buf, size_bytes());
            fun(begin(stage_buf) + offset);
		}
	}

	/// Copy range of values from device to host memory.
	template<class DstIter>
	auto rangeToHost(size_t offset_begin, size_t offset_end, DstIter dst_begin) const-> void {
		if(Base::isHostVisible()){
			auto copy_from = host_data();
            Base::invalidate_mapped_cache();
			std::copy(copy_from + offset_begin, copy_from + offset_end, dst_begin);
            unmap_host_data();
		} else {
			using std::begin; using std::end;
			auto stage_buf = HostArray<T, AllocDevice<properties::HostCached>>(Base::_dev
			                                                          , offset_end - offset_begin);
			copyBuf(Base::_dev, *this, stage_buf, size_bytes(), offset_begin, 0u);
			std::copy(begin(stage_buf), end(stage_buf), dst_begin);
		}
	}
	
	/// @return host container with a copy of array data.
	template<class C, typename=typename std::enable_if_t<vuh::traits::is_iterable<C>::value>>
	auto toHost() const-> C {
		auto ret = C(size());
		using std::begin;
		toHost(begin(ret));
		return ret;
	}

	/// @return number of elements
	auto size() const-> size_t {return _size;}
    using Base::size_bytes;

	/// doc me
	auto device_begin()-> ArrayIter<DeviceArray> { return ArrayIter<DeviceArray>(*this, 0); }
	auto device_begin() const-> ArrayIter<DeviceArray> { return ArrayIter<DeviceArray>(*this, 0); }

	/// doc me
	auto device_end()-> ArrayIter<DeviceArray> {return ArrayIter<DeviceArray>(*this, _size);}
	auto device_end() const-> ArrayIter<DeviceArray> {return ArrayIter<DeviceArray>(*this, _size);}
private: // helpers
	auto host_data()-> T* {
		assert(Base::isHostVisible());
        if (!ptr) {
            ptr = Base::template mapMemory<T>();
            Base::invalidate_mapped_cache();
        }
        return ptr;
	}

	auto host_data() const-> const T* {
		assert(Base::isHostVisible());
        if (!ptr) {
            ptr = Base::template mapMemory<T>();
            Base::invalidate_mapped_cache();
        }
        return ptr;
	}

    void unmap_host_data() const
    {
        if (Base::require_unmap_flush) {
            Base::unmapMemory();
            ptr = nullptr;
        }
    }
private: // data
	size_t _size; ///< number of elements. Actual allocated memory may be a bit bigger than necessary.
    mutable T* ptr = nullptr;
}; // class DeviceArray

/// doc me
template<class T, class Alloc>
auto device_begin(DeviceArray<T, Alloc>& array)-> ArrayIter<DeviceArray<T, Alloc>> {
	return array.device_begin();
}

/// doc me
template<class T, class Alloc>
auto device_end(DeviceArray<T, Alloc>& array)-> ArrayIter<DeviceArray<T, Alloc>> {
	return array.device_end();
}

} // namespace arr
} // namespace vuh
