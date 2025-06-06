## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2022-2025 Intel Corporation
## Copyright (c) 2022-2025 Georgia Institute of Technology
## ******************************************************************************

# TACOS library
add_library(tacos
    AllGather.cpp ${CMAKE_SOURCE_DIR}/include/tacos/AllGather.h
    Collective.cpp ${CMAKE_SOURCE_DIR}/include/tacos/Collective.h
    EventQueue.cpp ${CMAKE_SOURCE_DIR}/include/tacos/EventQueue.h
    Hypercube3D.cpp ${CMAKE_SOURCE_DIR}/include/tacos/Hypercube3D.h
    Mesh2D.cpp ${CMAKE_SOURCE_DIR}/include/tacos/Mesh2D.h
    TacosGreedy.cpp ${CMAKE_SOURCE_DIR}/include/tacos/TacosGreedy.h
    TacosNetwork.cpp ${CMAKE_SOURCE_DIR}/include/tacos/TacosNetwork.h
    Timer.cpp ${CMAKE_SOURCE_DIR}/include/tacos/Timer.h
    Topology.cpp ${CMAKE_SOURCE_DIR}/include/tacos/Topology.h
    Torus2D.cpp ${CMAKE_SOURCE_DIR}/include/tacos/Torus2D.h
    Torus3D.cpp ${CMAKE_SOURCE_DIR}/include/tacos/Torus3D.h
)
target_include_directories(tacos
    PUBLIC ${CMAKE_SOURCE_DIR}/include/tacos
)

# TACOS executable
add_executable(tacos_exec
    main.cpp
)
target_link_libraries(tacos_exec PRIVATE tacos)
set_target_properties(tacos_exec PROPERTIES OUTPUT_NAME tacos)
