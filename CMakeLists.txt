cmake_minimum_required(VERSION 3.16)
project(hackathon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Student solution
add_executable(solution hackathon.cpp solution.cpp)
target_link_libraries(solution m Threads::Threads)

# Baseline (the score to beat)
add_executable(baseline hackathon.cpp baseline.cpp)
target_link_libraries(baseline m Threads::Threads)

# Simulator without its main(), for the tools
add_library(hackathon_sim OBJECT hackathon.cpp)
target_compile_definitions(hackathon_sim PRIVATE main=hackathon_main)

# Shared by the tools below
add_library(common OBJECT common/replay.cpp common/problem_io.cpp common/score.cpp)
target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/common)

# baseline.cpp's solve() renamed, as the reference makespan
add_library(baseline_ref OBJECT baseline.cpp)
target_compile_definitions(baseline_ref PRIVATE solve=baseline_solve)

set(TOOL_OBJS $<TARGET_OBJECTS:common> $<TARGET_OBJECTS:baseline_ref>
              $<TARGET_OBJECTS:hackathon_sim>)

# Timeline dump for viz/index.html
add_executable(trace viz/trace.cpp solution.cpp ${TOOL_OBJS})
target_include_directories(trace PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/common)
target_compile_definitions(trace PRIVATE TRACE_OUT="${CMAKE_SOURCE_DIR}/viz/trace.js")
target_link_libraries(trace m Threads::Threads)

# Scores solution.cpp like the server, also on random problems
add_executable(check tools/check.cpp solution.cpp ${TOOL_OBJS})
target_include_directories(check PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/common)
target_compile_definitions(check PRIVATE
  PUBLIC_PROBLEMS="${CMAKE_SOURCE_DIR}/problems/public.txt"
  TRACE_OUT="${CMAKE_SOURCE_DIR}/viz/trace.js")
target_link_libraries(check m Threads::Threads)
