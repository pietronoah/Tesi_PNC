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
CMAKE_SOURCE_DIR = /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/build

# Include any dependencies generated for this target.
include CMakeFiles/ipopt_interface.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ipopt_interface.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ipopt_interface.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ipopt_interface.dir/flags.make

CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o: CMakeFiles/ipopt_interface.dir/flags.make
CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o: ../test_4_nlp.cpp
CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o: CMakeFiles/ipopt_interface.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o -MF CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o.d -o CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o -c /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/test_4_nlp.cpp

CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/test_4_nlp.cpp > CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.i

CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/test_4_nlp.cpp -o CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.s

# Object files for target ipopt_interface
ipopt_interface_OBJECTS = \
"CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o"

# External object files for target ipopt_interface
ipopt_interface_EXTERNAL_OBJECTS =

libipopt_interface.a: CMakeFiles/ipopt_interface.dir/test_4_nlp.cpp.o
libipopt_interface.a: CMakeFiles/ipopt_interface.dir/build.make
libipopt_interface.a: CMakeFiles/ipopt_interface.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libipopt_interface.a"
	$(CMAKE_COMMAND) -P CMakeFiles/ipopt_interface.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ipopt_interface.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ipopt_interface.dir/build: libipopt_interface.a
.PHONY : CMakeFiles/ipopt_interface.dir/build

CMakeFiles/ipopt_interface.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ipopt_interface.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ipopt_interface.dir/clean

CMakeFiles/ipopt_interface.dir/depend:
	cd /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4 /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4 /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/build /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/build /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi+Jinja+Autodiff/Reverse_mode/Test_4/build/CMakeFiles/ipopt_interface.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ipopt_interface.dir/depend

