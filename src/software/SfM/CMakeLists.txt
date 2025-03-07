
###
# Intrinsic image analysis and SfM_Data container initialization
###
add_executable(openMVG_main_SfMInit_ImageListing main_SfMInit_ImageListing.cpp)
target_link_libraries(openMVG_main_SfMInit_ImageListing
  PRIVATE
    openMVG_system
    openMVG_image
    openMVG_features
    openMVG_sfm
    openMVG_exif
    ${STLPLUS_LIBRARY}
)

#convert a v0.6 lists.txt file to the new sfm_data.X format
add_executable(openMVG_main_ConvertList main_ConvertList.cpp)
target_link_libraries(openMVG_main_ConvertList
  PRIVATE
    openMVG_system
    openMVG_features
    openMVG_sfm
    ${STLPLUS_LIBRARY}
)

# Installation rules
set_property(TARGET openMVG_main_SfMInit_ImageListing PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_SfMInit_ImageListing DESTINATION bin/)
set_property(TARGET openMVG_main_ConvertList PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_ConvertList DESTINATION bin/)

###
# Add executable that computes:
# - openMVG_main_ComputeFeatures: features and descriptors
# - openMVG_main_ComputeMatches: putative matches + geometric filtered matches
###

add_executable(openMVG_main_ComputeFeatures main_ComputeFeatures.cpp)
target_link_libraries(openMVG_main_ComputeFeatures
  PRIVATE
    openMVG_system
    openMVG_image
    openMVG_features
    openMVG_multiview
    openMVG_sfm
    ${STLPLUS_LIBRARY}
    vlsift
)
if (MSVC)
  set_target_properties(openMVG_main_ComputeFeatures PROPERTIES COMPILE_FLAGS "/bigobj")
endif (MSVC)

add_executable(openMVG_main_ListMatchingPairs main_ListMatchingPairs.cpp)
target_link_libraries(openMVG_main_ListMatchingPairs
  PRIVATE
    openMVG_features
    openMVG_multiview
    openMVG_sfm
    openMVG_system
    ${STLPLUS_LIBRARY}
)

add_executable( openMVG_main_PairGenerator main_PairGenerator.cpp )
target_link_libraries( openMVG_main_PairGenerator
  PRIVATE
    openMVG_features
    openMVG_multiview
    openMVG_sfm
    openMVG_system
    ${STLPLUS_LIBRARY}
)

# - convert matches from one format (bin, txt) to the other format
#
add_executable(openMVG_main_ConvertMatches main_ConvertMatches.cpp)
target_link_libraries(openMVG_main_ConvertMatches
  openMVG_system
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_benchANN main_benchANN.cpp)
target_link_libraries(openMVG_main_benchANN
  PRIVATE
    openMVG_features
    openMVG_matching_image_collection
    openMVG_multiview
    openMVG_sfm
    openMVG_system
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_ComputeVLAD main_ComputeVLAD.cpp)
target_link_libraries(openMVG_main_ComputeVLAD
  PRIVATE
    openMVG_features
    openMVG_sfm
    openMVG_system
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_MatchingRetrievalQuality main_MatchingRetrievalQuality.cpp)
target_link_libraries(openMVG_main_MatchingRetrievalQuality
  PRIVATE
    openMVG_features
    openMVG_matching_image_collection
    openMVG_sfm
    openMVG_system
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_ComputeMatches main_ComputeMatches.cpp)
target_link_libraries(openMVG_main_ComputeMatches
  PRIVATE
    openMVG_graph
    openMVG_features
    openMVG_matching_image_collection
    openMVG_multiview
    openMVG_sfm
    openMVG_system
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_GeometricFilter main_GeometricFilter.cpp)
target_link_libraries(openMVG_main_GeometricFilter
  PRIVATE
    openMVG_graph
    openMVG_features
    openMVG_matching_image_collection
    openMVG_multiview
    openMVG_sfm
    openMVG_system
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_MatchesToTracks main_MatchesToTracks.cpp)
target_link_libraries(openMVG_main_MatchesToTracks
  PRIVATE
    openMVG_system
    openMVG_image
    openMVG_features
    openMVG_sfm
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_ScaleTestField main_ScaleTestField.cpp)
target_link_libraries(openMVG_main_ScaleTestField
  openMVG_system
  openMVG_image
  openMVG_sfm
  ${STLPLUS_LIBRARY}
  ${OpenCV_LIBS}
)
target_include_directories(openMVG_main_ScaleTestField PRIVATE ${OpenCV_INCLUDE_DIRS})

# Installation rules
set_property( TARGET openMVG_main_ListMatchingPairs PROPERTY FOLDER OpenMVG/software )
set_property( TARGET openMVG_main_ComputeFeatures   PROPERTY FOLDER OpenMVG/software )
set_property( TARGET openMVG_main_PairGenerator     PROPERTY FOLDER OpenMVG/software )
set_property( TARGET openMVG_main_ComputeVLAD       PROPERTY FOLDER OpenMVG/software )
set_property( TARGET openMVG_main_ComputeMatches    PROPERTY FOLDER OpenMVG/software )
set_property( TARGET openMVG_main_GeometricFilter   PROPERTY FOLDER OpenMVG/software )
set_property( TARGET openMVG_main_MatchesToTracks   PROPERTY FOLDER OpenMVG/software )
set_property( TARGET openMVG_main_ScaleTestField    PROPERTY FOLDER OpenMVG/software )

install( TARGETS openMVG_main_ListMatchingPairs DESTINATION bin/ )
install( TARGETS openMVG_main_ComputeFeatures   DESTINATION bin/ )
install( TARGETS openMVG_main_PairGenerator     DESTINATION bin/ )
install( TARGETS openMVG_main_ComputeVLAD       DESTINATION bin/ )
install( TARGETS openMVG_main_ComputeMatches    DESTINATION bin/ )
install( TARGETS openMVG_main_GeometricFilter   DESTINATION bin/ )
install( TARGETS openMVG_main_MatchesToTracks   DESTINATION bin/ )
install( TARGETS openMVG_main_ScaleTestField    DESTINATION bin/ )

###
# SfM Pipelines & SfM Data format tools, ...
###
add_executable(openMVG_main_SfM main_SfM.cpp)
target_link_libraries(openMVG_main_SfM
  PRIVATE
    openMVG_system
    openMVG_image
    openMVG_features
    openMVG_sfm
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_ConvertSfM_DataFormat main_ConvertSfM_DataFormat.cpp)
target_link_libraries(openMVG_main_ConvertSfM_DataFormat
  PRIVATE
    openMVG_system
    openMVG_features
    openMVG_sfm
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_FrustumFiltering main_FrustumFiltering.cpp)
target_link_libraries(openMVG_main_FrustumFiltering
  PRIVATE
    openMVG_system
    openMVG_features
    openMVG_sfm
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_ComputeStructureFromKnownPoses main_ComputeStructureFromKnownPoses.cpp)
target_include_directories(openMVG_main_ComputeStructureFromKnownPoses
  PRIVATE
    ${CERES_INCLUDE_DIRS}
)
target_link_libraries(openMVG_main_ComputeStructureFromKnownPoses
  PRIVATE
    openMVG_system
    openMVG_features
    openMVG_sfm
    ${STLPLUS_LIBRARY}
    ${CERES_LIBRARIES}
)

add_executable(openMVG_main_ComputeSfM_DataColor main_ComputeSfM_DataColor.cpp)
target_link_libraries(openMVG_main_ComputeSfM_DataColor
  PRIVATE
    openMVG_system
    openMVG_image
    openMVG_features
    openMVG_sfm
    ${STLPLUS_LIBRARY}
)

add_executable(openMVG_main_PointsFiltering main_PointsFiltering.cpp)
target_link_libraries(openMVG_main_PointsFiltering
  PRIVATE
    openMVG_system
    openMVG_sfm
    ${STLPLUS_LIBRARY}
)
target_include_directories(openMVG_main_PointsFiltering PRIVATE ${EIGEN_INCLUDE_DIRS})

add_executable(openMVG_main_ChangeLocalOrigin main_ChangeLocalOrigin.cpp)
target_link_libraries(openMVG_main_ChangeLocalOrigin
  PRIVATE
    openMVG_system
    openMVG_image
    openMVG_features
    openMVG_sfm
    ${STLPLUS_LIBRARY}
)

# Installation rules
set_property(TARGET openMVG_main_SfM PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_SfM DESTINATION bin/)
set_property(TARGET openMVG_main_ConvertSfM_DataFormat PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_ConvertSfM_DataFormat DESTINATION bin/)
set_property(TARGET openMVG_main_FrustumFiltering PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_FrustumFiltering DESTINATION bin/)
set_property(TARGET openMVG_main_ComputeStructureFromKnownPoses PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_ComputeStructureFromKnownPoses DESTINATION bin/)
set_property(TARGET openMVG_main_ComputeSfM_DataColor PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_ComputeSfM_DataColor DESTINATION bin/)
set_property(TARGET openMVG_main_PointsFiltering PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_PointsFiltering DESTINATION bin/)
set_property(TARGET openMVG_main_ChangeLocalOrigin PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_ChangeLocalOrigin DESTINATION bin/)

# SplitMatchFileIntoMatchFiles
add_executable(openMVG_main_SplitMatchFileIntoMatchFiles main_SplitMatchFileIntoMatchFiles.cpp)
target_link_libraries(openMVG_main_SplitMatchFileIntoMatchFiles
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# Installation rules
set_property(TARGET openMVG_main_SplitMatchFileIntoMatchFiles PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_SplitMatchFileIntoMatchFiles DESTINATION bin/)

# -
# Use openCV Feature detector to describe the images
# -
if(OpenMVG_USE_OPENCV)
  add_executable(openMVG_main_ComputeFeatures_OpenCV
    main_ComputeFeatures_OpenCV.cpp)
  target_link_libraries(openMVG_main_ComputeFeatures_OpenCV
    openMVG_system
    openMVG_image
    openMVG_features
    openMVG_multiview
    openMVG_sfm
    ${STLPLUS_LIBRARY}
    vlsift
    ${OpenCV_LIBS})
  target_include_directories(openMVG_main_ComputeFeatures_OpenCV PRIVATE ${OpenCV_INCLUDE_DIRS})

  if(OpenMVG_USE_OCVSIFT)
    target_compile_definitions(openMVG_main_ComputeFeatures_OpenCV PRIVATE USE_OCVSIFT)
  endif(OpenMVG_USE_OCVSIFT)

  set_property(TARGET openMVG_main_ComputeFeatures_OpenCV PROPERTY FOLDER OpenMVG/software)
  set_property(TARGET openMVG_main_ComputeFeatures_OpenCV PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
  install(TARGETS openMVG_main_ComputeFeatures_OpenCV DESTINATION bin/)
endif(OpenMVG_USE_OPENCV)


###
# SfM Research tools
#
# - Quality comparison against a GT camera path (MultiView Evaluation dataset)
###
add_executable(openMVG_main_evalQuality main_evalQuality.cpp)
target_link_libraries(openMVG_main_evalQuality
  openMVG_features
  openMVG_image
  openMVG_sfm
  openMVG_system
  ${STLPLUS_LIBRARY}
)

#installation rules
set_property(TARGET openMVG_main_evalQuality PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_evalQuality DESTINATION bin/)

##
# Export tutorial with valid path thanks to a configuration file
##
set(OPENMVG_SOFTWARE_SFM_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})
if(MSVC)
  set(OPENMVG_SOFTWARE_SFM_BUILD_DIR "${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE}")
else()
  if( APPLE )
    if( CMAKE_GENERATOR MATCHES "Xcode" )
      set(OPENMVG_SOFTWARE_SFM_BUILD_DIR "${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE}")
    else()
      set(OPENMVG_SOFTWARE_SFM_BUILD_DIR "${EXECUTABLE_OUTPUT_PATH}")
    endif()
  else()
    # Linux
    set(OPENMVG_SOFTWARE_SFM_BUILD_DIR ${EXECUTABLE_OUTPUT_PATH})
  endif()
endif()

set(OPENMVG_TUTORIAL_IN_FILE "${CMAKE_CURRENT_LIST_DIR}/tutorial_demo.py.in")
configure_file(${OPENMVG_TUTORIAL_IN_FILE}
    "${CMAKE_CURRENT_BINARY_DIR}/tutorial_demo.py" @ONLY)

set(OPENMVG_TUTORIAL_IN_FILE "${CMAKE_CURRENT_LIST_DIR}/SfM_GlobalPipeline.py.in")
configure_file(${OPENMVG_TUTORIAL_IN_FILE}
    "${CMAKE_CURRENT_BINARY_DIR}/SfM_GlobalPipeline.py" @ONLY)

set(OPENMVG_TUTORIAL_IN_FILE "${CMAKE_CURRENT_LIST_DIR}/SfM_SequentialPipeline.py.in")
configure_file(${OPENMVG_TUTORIAL_IN_FILE}
    "${CMAKE_CURRENT_BINARY_DIR}/SfM_SequentialPipeline.py" @ONLY)

add_subdirectory(clustering)
add_subdirectory(export)
add_subdirectory(import)
