add_executable(run_tests
  test_fd_interpolate.cpp
  test_fd_partial_derivative.cpp
  test_fd_grad.cpp
)

target_link_libraries(run_tests
  core  
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_tests)
