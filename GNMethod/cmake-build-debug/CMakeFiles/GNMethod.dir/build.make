# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/alexis/Code/ComputationalGraphics/GNMethod

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/GNMethod.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/GNMethod.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GNMethod.dir/flags.make

CMakeFiles/GNMethod.dir/main.cpp.o: CMakeFiles/GNMethod.dir/flags.make
CMakeFiles/GNMethod.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GNMethod.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GNMethod.dir/main.cpp.o -c /Users/alexis/Code/ComputationalGraphics/GNMethod/main.cpp

CMakeFiles/GNMethod.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GNMethod.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alexis/Code/ComputationalGraphics/GNMethod/main.cpp > CMakeFiles/GNMethod.dir/main.cpp.i

CMakeFiles/GNMethod.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GNMethod.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alexis/Code/ComputationalGraphics/GNMethod/main.cpp -o CMakeFiles/GNMethod.dir/main.cpp.s

CMakeFiles/GNMethod.dir/Solverxxxx.cpp.o: CMakeFiles/GNMethod.dir/flags.make
CMakeFiles/GNMethod.dir/Solverxxxx.cpp.o: ../Solverxxxx.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/GNMethod.dir/Solverxxxx.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GNMethod.dir/Solverxxxx.cpp.o -c /Users/alexis/Code/ComputationalGraphics/GNMethod/Solverxxxx.cpp

CMakeFiles/GNMethod.dir/Solverxxxx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GNMethod.dir/Solverxxxx.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alexis/Code/ComputationalGraphics/GNMethod/Solverxxxx.cpp > CMakeFiles/GNMethod.dir/Solverxxxx.cpp.i

CMakeFiles/GNMethod.dir/Solverxxxx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GNMethod.dir/Solverxxxx.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alexis/Code/ComputationalGraphics/GNMethod/Solverxxxx.cpp -o CMakeFiles/GNMethod.dir/Solverxxxx.cpp.s

CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.o: CMakeFiles/GNMethod.dir/flags.make
CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.o: ../ResidualFunctionxxxx.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.o -c /Users/alexis/Code/ComputationalGraphics/GNMethod/ResidualFunctionxxxx.cpp

CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alexis/Code/ComputationalGraphics/GNMethod/ResidualFunctionxxxx.cpp > CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.i

CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alexis/Code/ComputationalGraphics/GNMethod/ResidualFunctionxxxx.cpp -o CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.s

# Object files for target GNMethod
GNMethod_OBJECTS = \
"CMakeFiles/GNMethod.dir/main.cpp.o" \
"CMakeFiles/GNMethod.dir/Solverxxxx.cpp.o" \
"CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.o"

# External object files for target GNMethod
GNMethod_EXTERNAL_OBJECTS =

GNMethod: CMakeFiles/GNMethod.dir/main.cpp.o
GNMethod: CMakeFiles/GNMethod.dir/Solverxxxx.cpp.o
GNMethod: CMakeFiles/GNMethod.dir/ResidualFunctionxxxx.cpp.o
GNMethod: CMakeFiles/GNMethod.dir/build.make
GNMethod: /usr/local/anaconda3/lib/libopencv_stitching.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_superres.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_videostab.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_aruco.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_bgsegm.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_bioinspired.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_ccalib.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_dnn_objdetect.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_dpm.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_face.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_freetype.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_fuzzy.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_hdf.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_hfs.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_img_hash.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_line_descriptor.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_optflow.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_reg.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_rgbd.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_saliency.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_stereo.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_structured_light.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_surface_matching.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_tracking.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_xfeatures2d.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_ximgproc.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_xobjdetect.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_xphoto.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_shape.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_photo.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_calib3d.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_phase_unwrapping.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_video.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_datasets.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_plot.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_text.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_dnn.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_features2d.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_flann.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_highgui.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_ml.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_videoio.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_imgcodecs.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_objdetect.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_imgproc.3.4.2.dylib
GNMethod: /usr/local/anaconda3/lib/libopencv_core.3.4.2.dylib
GNMethod: CMakeFiles/GNMethod.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable GNMethod"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GNMethod.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GNMethod.dir/build: GNMethod

.PHONY : CMakeFiles/GNMethod.dir/build

CMakeFiles/GNMethod.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GNMethod.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GNMethod.dir/clean

CMakeFiles/GNMethod.dir/depend:
	cd /Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/alexis/Code/ComputationalGraphics/GNMethod /Users/alexis/Code/ComputationalGraphics/GNMethod /Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug /Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug /Users/alexis/Code/ComputationalGraphics/GNMethod/cmake-build-debug/CMakeFiles/GNMethod.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GNMethod.dir/depend
