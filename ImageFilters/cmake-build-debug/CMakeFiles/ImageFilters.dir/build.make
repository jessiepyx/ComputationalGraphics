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
CMAKE_SOURCE_DIR = /Users/alexis/Code/ComputationalGraphics/ImageFilters

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/alexis/Code/ComputationalGraphics/ImageFilters/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/ImageFilters.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ImageFilters.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ImageFilters.dir/flags.make

CMakeFiles/ImageFilters.dir/main.cpp.o: CMakeFiles/ImageFilters.dir/flags.make
CMakeFiles/ImageFilters.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/alexis/Code/ComputationalGraphics/ImageFilters/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ImageFilters.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ImageFilters.dir/main.cpp.o -c /Users/alexis/Code/ComputationalGraphics/ImageFilters/main.cpp

CMakeFiles/ImageFilters.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ImageFilters.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alexis/Code/ComputationalGraphics/ImageFilters/main.cpp > CMakeFiles/ImageFilters.dir/main.cpp.i

CMakeFiles/ImageFilters.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ImageFilters.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alexis/Code/ComputationalGraphics/ImageFilters/main.cpp -o CMakeFiles/ImageFilters.dir/main.cpp.s

# Object files for target ImageFilters
ImageFilters_OBJECTS = \
"CMakeFiles/ImageFilters.dir/main.cpp.o"

# External object files for target ImageFilters
ImageFilters_EXTERNAL_OBJECTS =

ImageFilters: CMakeFiles/ImageFilters.dir/main.cpp.o
ImageFilters: CMakeFiles/ImageFilters.dir/build.make
ImageFilters: /usr/local/anaconda3/lib/libopencv_stitching.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_superres.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_videostab.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_aruco.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_bgsegm.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_bioinspired.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_ccalib.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_dnn_objdetect.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_dpm.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_face.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_freetype.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_fuzzy.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_hdf.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_hfs.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_img_hash.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_line_descriptor.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_optflow.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_reg.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_rgbd.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_saliency.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_stereo.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_structured_light.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_surface_matching.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_tracking.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_xfeatures2d.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_ximgproc.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_xobjdetect.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_xphoto.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_shape.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_photo.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_calib3d.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_phase_unwrapping.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_video.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_datasets.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_plot.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_text.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_dnn.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_features2d.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_flann.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_highgui.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_ml.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_videoio.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_imgcodecs.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_objdetect.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_imgproc.3.4.2.dylib
ImageFilters: /usr/local/anaconda3/lib/libopencv_core.3.4.2.dylib
ImageFilters: CMakeFiles/ImageFilters.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/alexis/Code/ComputationalGraphics/ImageFilters/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ImageFilters"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ImageFilters.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ImageFilters.dir/build: ImageFilters

.PHONY : CMakeFiles/ImageFilters.dir/build

CMakeFiles/ImageFilters.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ImageFilters.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ImageFilters.dir/clean

CMakeFiles/ImageFilters.dir/depend:
	cd /Users/alexis/Code/ComputationalGraphics/ImageFilters/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/alexis/Code/ComputationalGraphics/ImageFilters /Users/alexis/Code/ComputationalGraphics/ImageFilters /Users/alexis/Code/ComputationalGraphics/ImageFilters/cmake-build-debug /Users/alexis/Code/ComputationalGraphics/ImageFilters/cmake-build-debug /Users/alexis/Code/ComputationalGraphics/ImageFilters/cmake-build-debug/CMakeFiles/ImageFilters.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ImageFilters.dir/depend

