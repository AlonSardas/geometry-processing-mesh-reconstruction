add_executable(run_tests
  test_fd_interpolate.cpp
  test_partial_derivative.cpp
)

target_link_libraries(run_tests
  core  
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_tests)
