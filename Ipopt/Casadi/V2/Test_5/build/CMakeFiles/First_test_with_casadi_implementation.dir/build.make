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
CMAKE_SOURCE_DIR = /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/build

# Include any dependencies generated for this target.
include CMakeFiles/First_test_with_casadi_implementation.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/First_test_with_casadi_implementation.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/First_test_with_casadi_implementation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/First_test_with_casadi_implementation.dir/flags.make

CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o: CMakeFiles/First_test_with_casadi_implementation.dir/flags.make
CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o: ../hs071_main.cpp
CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o: CMakeFiles/First_test_with_casadi_implementation.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o -MF CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o.d -o CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o -c /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/hs071_main.cpp

CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/hs071_main.cpp > CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.i

CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/hs071_main.cpp -o CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.s

# Object files for target First_test_with_casadi_implementation
First_test_with_casadi_implementation_OBJECTS = \
"CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o"

# External object files for target First_test_with_casadi_implementation
First_test_with_casadi_implementation_EXTERNAL_OBJECTS =

First_test_with_casadi_implementation: CMakeFiles/First_test_with_casadi_implementation.dir/hs071_main.cpp.o
First_test_with_casadi_implementation: CMakeFiles/First_test_with_casadi_implementation.dir/build.make
First_test_with_casadi_implementation: libipopt_interface.a
First_test_with_casadi_implementation: CMakeFiles/First_test_with_casadi_implementation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable First_test_with_casadi_implementation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/First_test_with_casadi_implementation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/First_test_with_casadi_implementation.dir/build: First_test_with_casadi_implementation
.PHONY : CMakeFiles/First_test_with_casadi_implementation.dir/build

CMakeFiles/First_test_with_casadi_implementation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/First_test_with_casadi_implementation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/First_test_with_casadi_implementation.dir/clean

CMakeFiles/First_test_with_casadi_implementation.dir/depend:
	cd /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5 /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5 /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/build /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/build /Users/pietronoahcrestaz/Documents/Pietro/UniTn/Tesi/Code/Ipopt/Casadi/Test_integrazione_CasADi/V2/Test_5/build/CMakeFiles/First_test_with_casadi_implementation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/First_test_with_casadi_implementation.dir/depend

