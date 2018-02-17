project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/particle_filter.cpp src/main.cpp)


if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

if (WIN32)
    set(VCPKG_INSTALL ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET})
    message("Using vcpkg_install ${VCPKG_INSTALL}")
    message("Using Windows Target Platform Version: ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
    include_directories(${VCPKG_INSTALL}/include)
    link_directories(${VCPKG_INSTALL}/lib)
endif()

add_executable(particle_filter ${sources})

if (NOT WIN32)
    target_link_libraries(particle_filter z ssl uv uWS)
else()
    target_link_libraries(particle_filter ssleay32 libuv uWS ws2_32)
    target_link_libraries(particle_filter optimized zlib )
    target_link_libraries(particle_filter debug zlibd )
endif()

