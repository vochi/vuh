#pragma once

#include "basicArray.hpp"
#include "arrayIter.hpp"

#include <algorithm>

namespace vuh {
namespace arr {

/// Array with host data exchange interface suitable for memory allocated in host-visible space.
/// Memory allocation and underlying buffer creation is managed by allocator defined by a template parameter.
/// Such allocator is expected to allocate memory in host-visible GPU memory, mappable
/// for host access.
/// Provides Forward iterator + random-access interface.
/// Memory remains mapped during the whole lifetime of an object of this type.
///
/// !!!!!
/// Flush/invalidate is a user responsibility!!!
/// !!!!!
template<class T, class Alloc>
class HostArray: public BasicArray<Alloc> {
	using Base = BasicArray<Alloc>;
public:
	using value_type = T;
    HostArray() : _data(nullptr) {}
	/// Construct object of the class on given device.
	/// Memory is not initialized with any data.
	HostArray(vuh::Device& device  ///< device to create array on
	          , size_t n_elements  ///< number of elements
	          , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	          , vk::BufferUsageFlags flags_buffer={}    ///< additional (to defined by allocator) buffer usage flags
	          )
	   : BasicArray<Alloc>(device, n_elements*sizeof(T), flags_memory, flags_buffer)
       , _data(Base::template mapMemory<T>())
	   , _size(n_elements)
	{}

	/// Construct array on given device and initialize with a provided value.
	HostArray( vuh::Device& device ///< device to create array on
	         , size_t n_elements   ///< number of elements
	         , T value             ///< initializer value
	         , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	         , vk::BufferUsageFlags flags_buffer={}	   ///< additional (to defined by allocator) buffer usage flags
	         )
	   : HostArray(device, n_elements, flags_memory, flags_buffer)
	{
		std::fill_n(begin(), n_elements, value);
        Base::flush_mapped_writes();
        unmap_host_data();
	}

	/// Construct array on given device and initialize it from range of values
	template<class It1, class It2>
	HostArray(vuh::Device& device ///< device to create array on
	         , It1 begin          ///< beginning of initialization range
	         , It2 end            ///< end (one past end) of initialization range
	         , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	         , vk::BufferUsageFlags flags_buffer={}    ///< additional (to defined by allocator) buffer usage flags
	         )
	   : HostArray(device, std::distance(begin, end), flags_memory, flags_buffer)
	{
		std::copy(begin, end, this->begin());
        Base::flush_mapped_writes();
        unmap_host_data();
	}
    /// Construct array on given device and call fun to fill it
	template<typename F>
	HostArray(vuh::Device& device ///< device to create array on
             , size_t size
             , F&& fun
	         , vk::MemoryPropertyFlags flags_memory={} ///< additional (to defined by allocator) memory usage flags
	         , vk::BufferUsageFlags flags_buffer={}    ///< additional (to defined by allocator) buffer usage flags
	         )
	   : HostArray(device, size, flags_memory, flags_buffer)
	{
		fun(this->begin());
        Base::flush_mapped_writes();
        unmap_host_data();
	}
    /// Construct array on given device and initialize it from range of values
    template<class It1, class It2, typename F>
    HostArray(vuh::Device& device ///< device to create array on
             , It1 begin          ///< beginning of initialization range
             , It2 end            ///< end (one past end) of initialization range
              , F&& fun
             , std::enable_if_t<!std::is_same_v<vk::MemoryPropertyFlags, It2> && !std::is_same_v<vk::BufferUsageFlags, F>, vk::MemoryPropertyFlags> flags_memory={} ///< additional (to defined by allocator) memory usage flags
             , vk::BufferUsageFlags flags_buffer={}    ///< additional (to defined by allocator) buffer usage flags
             )
       : HostArray(device, std::distance(begin, end), flags_memory, flags_buffer)
    {
        std::transform(begin, end, this->begin(), fun);
        Base::flush_mapped_writes();
        unmap_host_data();
    }

   /// Move constructor.
   HostArray(HostArray&& o): Base(std::move(o)), _data(o._data), _size(o._size) {o._data = nullptr;}
	/// Move operator.
   auto operator=(HostArray&& o)-> HostArray& {
		this->swap(o);
		return *this;
   }
   
   /// Destroy array, and release all associated resources.
   ~HostArray() noexcept {
      if(_data) {
         Base::unmapMemory();
      }
   }

	/// doc me
	auto swap(HostArray& o) noexcept-> void {
		using std::swap;
		swap(static_cast<Base&>(*this), static_cast<Base&>(o));
      swap(_data, o._data);
      swap(_size, o._size);
	}

   
   /// Host-accesible iterator to beginning of array data
	auto begin()-> value_type* { return host_data(); }
	auto begin() const-> const value_type* { return host_data(); }
   
	/// Pointer (host) to beginning of array data
	auto data()-> T* {return host_data();}
	auto data() const-> const T* {return host_data();}

   /// Host-accessible iterator to the end (one past the last element) of array data.
	auto end()-> value_type* { return begin() + size(); }
	auto end() const-> const value_type* { return begin() + size(); }

	///
	auto device_begin()-> ArrayIter<HostArray> { return ArrayIter<HostArray>(*this, 0);}
	auto device_begin() const-> ArrayIter<HostArray> { return ArrayIter<HostArray>(*this, 0);}
	friend auto device_begin(HostArray& a)-> ArrayIter<HostArray> {return a.device_begin();}
	
	///
	auto device_end()-> ArrayIter<HostArray> {return ArrayIter<HostArray>(*this, _size);}
	auto device_end() const-> ArrayIter<HostArray> {return ArrayIter<HostArray>(*this, _size);}
	friend auto device_end(HostArray& a)-> ArrayIter<HostArray> {return a.device_end();}

   /// Element access operator (host-side).
   auto operator[](size_t i)-> T& { return *(begin() + i);}
   auto operator[](size_t i) const-> T { return *(begin() + i);}
   
   /// @return number of elements
   auto size() const-> size_t {return _size;}

   using Base::size_bytes;

   void unmap_host_data() const
   {
       if (Base::require_unmap_flush) {
           Base::unmapMemory();
           _data = nullptr;
       }
   }

private:
   auto host_data()-> T* {
       assert(Base::isHostVisible());
       if (!_data) {
           _data = Base::template mapMemory<T>();
           Base::invalidate_mapped_cache();
       }
       return _data;
   }

   auto host_data() const-> const T* {
       assert(Base::isHostVisible());
       if (!_data) {
           _data = Base::template mapMemory<T>();
           Base::invalidate_mapped_cache();
       }
       return _data;
   }

private: // data
   mutable T* _data;       ///< host accessible pointer to the beginning of corresponding memory chunk.
   size_t _size;  ///< Number of elements. Actual allocated memory may be a bit bigger then necessary.
}; // class HostArray
} // namespace arr
} // namespace vuh
