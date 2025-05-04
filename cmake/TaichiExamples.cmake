cmake_minimum_required(VERSION 3.0)

# set(EXAMPLES_NAME taichi_cpp_examples)

# file(GLOB_RECURSE TAICHI_EXAMPLES_SOURCE
# "cpp_examples/main.cpp"
# "cpp_examples/run_snode.cpp"
# "cpp_examples/run_snode_init_builder.cpp"
# "cpp_examples/run_snode_ret_mine.cpp"
# "cpp_examples/run_snode_ret_builder.cpp"
# "cpp_examples/autograd.cpp"
# "cpp_examples/aot_save.cpp"
# )

function(add_taichi_example NAME)
set(TARGET_NAME "cpp_examples_${NAME}")
set(SOURCE_FILE "cpp_examples/${NAME}.cpp")
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
  if (WIN32)
    # Output the executable to build/ instead of build/Debug/...
    set(TARGET_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TARGET_OUTPUT_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${TARGET_OUTPUT_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${TARGET_OUTPUT_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${TARGET_OUTPUT_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${TARGET_OUTPUT_DIR})
  endif()
  target_link_libraries(${TARGET_NAME} PRIVATE taichi_core)
  target_include_directories(${TARGET_NAME}
    PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/external/spdlog/include
      ${PROJECT_SOURCE_DIR}/external/eigen
  )
  if (TI_WITH_METAL)
  target_link_libraries(${TARGET_NAME} PRIVATE
    metal_program_impl
  )
  endif()

  if (TI_WITH_VULKAN OR TI_WITH_OPENGL OR TI_WITH_METAL)
    target_link_libraries(${TARGET_NAME} PRIVATE gfx_runtime)
  endif()

  if (TI_WITH_VULKAN)
    target_link_libraries(${TARGET_NAME} PRIVATE vulkan_rhi)
  endif()

  if (TI_WITH_OPENGL)
    target_link_libraries(${TARGET_NAME} PRIVATE opengl_rhi)
  endif()

  if (TI_WITH_METAL)
    target_link_libraries(${TARGET_NAME} PRIVATE metal_rhi)
  endif()
endfunction()

add_taichi_example(run_init)
add_taichi_example(run_ret)
add_taichi_example(run_ret_w_block)
add_taichi_example(run_ext)
add_taichi_example(scratchpad)
add_taichi_example(autograd)
add_taichi_example(aot_save)


# add_executable(${EXAMPLES_NAME} ${TAICHI_EXAMPLES_SOURCE})
# if (WIN32)
#     # Output the executable to build/ instead of build/Debug/...
#     set(EXAMPLES_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
#     set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLES_OUTPUT_DIR})
#     set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXAMPLES_OUTPUT_DIR})
#     set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXAMPLES_OUTPUT_DIR})
#     set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${EXAMPLES_OUTPUT_DIR})
#     set_target_properties(${EXAMPLES_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${EXAMPLES_OUTPUT_DIR})
# endif()

# target_link_libraries(${EXAMPLES_NAME} PRIVATE taichi_core)

# if (TI_WITH_METAL)
#   target_link_libraries(${EXAMPLES_NAME} PRIVATE
#     metal_program_impl
#   )
# endif()

# if (TI_WITH_VULKAN OR TI_WITH_OPENGL OR TI_WITH_METAL)
#   target_link_libraries(${EXAMPLES_NAME} PRIVATE gfx_runtime)
# endif()

# if (TI_WITH_VULKAN)
#   target_link_libraries(${EXAMPLES_NAME} PRIVATE vulkan_rhi)
# endif()

# if (TI_WITH_OPENGL)
#   target_link_libraries(${EXAMPLES_NAME} PRIVATE opengl_rhi)
# endif()

# if (TI_WITH_METAL)
#   target_link_libraries(${EXAMPLES_NAME} PRIVATE metal_rhi)
# endif()

# # TODO 4832: be specific on the header dependencies here, e.g., ir
# target_include_directories(${EXAMPLES_NAME}
#   PRIVATE
#     ${PROJECT_SOURCE_DIR}
#     ${PROJECT_SOURCE_DIR}/external/spdlog/include
#     ${PROJECT_SOURCE_DIR}/external/eigen
#   )
