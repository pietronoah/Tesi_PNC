# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/build

# Include any dependencies generated for this target.
include CMakeFiles/test_2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_2.dir/flags.make

CMakeFiles/test_2.dir/cpp_example.cpp.o: CMakeFiles/test_2.dir/flags.make
CMakeFiles/test_2.dir/cpp_example.cpp.o: ../cpp_example.cpp
CMakeFiles/test_2.dir/cpp_example.cpp.o: CMakeFiles/test_2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_2.dir/cpp_example.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_2.dir/cpp_example.cpp.o -MF CMakeFiles/test_2.dir/cpp_example.cpp.o.d -o CMakeFiles/test_2.dir/cpp_example.cpp.o -c /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/cpp_example.cpp

CMakeFiles/test_2.dir/cpp_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_2.dir/cpp_example.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/cpp_example.cpp > CMakeFiles/test_2.dir/cpp_example.cpp.i

CMakeFiles/test_2.dir/cpp_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_2.dir/cpp_example.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/cpp_example.cpp -o CMakeFiles/test_2.dir/cpp_example.cpp.s

# Object files for target test_2
test_2_OBJECTS = \
"CMakeFiles/test_2.dir/cpp_example.cpp.o"

# External object files for target test_2
test_2_EXTERNAL_OBJECTS =

test_2: CMakeFiles/test_2.dir/cpp_example.cpp.o
test_2: CMakeFiles/test_2.dir/build.make
test_2: libipopt_interface.a
test_2: CMakeFiles/test_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_2.dir/build: test_2
.PHONY : CMakeFiles/test_2.dir/build

CMakeFiles/test_2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_2.dir/clean

CMakeFiles/test_2.dir/depend:
	cd /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2 /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2 /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/build /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/build /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Test/Ipopt/C++/Test_2/build/CMakeFiles/test_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_2.dir/depend

