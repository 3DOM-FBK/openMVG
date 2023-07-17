#include "openMVG/image/image_io.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/multiview/triangulation_nview.hpp"
#include "openMVG/sfm/sfm_data_triangulation.hpp"
#include "openMVG/geometry/rigid_transformation3D_srt.hpp"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"

#include "third_party/cmdLine/cmdLine.h"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include "openMVG/stl/split.hpp"

#include <fstream>

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::cameras;
using namespace openMVG::sfm;

void gcpRegister(SfM_Data & sfm_data)
{
  if (sfm_data.control_points.size() < 3)
  {
    std::cout << "Not enough control points (min. 3 required).\n\n";
    return;
  }

  //---
  // registration (coarse):
  // - compute the 3D points corresponding to the control point observation for the SfM scene
  // - compute a coarse registration between the controls points & the triangulated point
  // - transform the scene according the found transformation
  //---

  for (int repeat = 0; repeat < 2; repeat++)
  {
    std::map<IndexT, Vec3> map_control_points, map_triangulated;
    
    std::map<IndexT, double> map_triangulation_errors;
    for (const auto & control_point_it : sfm_data.control_points)
    {
      const Landmark & landmark = control_point_it.second;
      //Triangulate the observations:
      const Observations & obs = landmark.obs;

      if (obs.size() < 2)
      {
          std::cout << "Control point must be defined in at least 2 pictures. Skipping.\n\n";
          continue;
      }

      std::vector<Vec3> bearing;
      std::vector<Mat34> poses;
      bearing.reserve(obs.size());
      poses.reserve(obs.size());

      for (const auto & obs_it : obs)
      {
        const View * view = sfm_data.views.at(obs_it.first).get();
        if (!sfm_data.IsPoseAndIntrinsicDefined(view))
          continue;
        const openMVG::cameras::IntrinsicBase * cam = sfm_data.GetIntrinsics().at(view->id_intrinsic).get();
        const openMVG::geometry::Pose3 pose = sfm_data.GetPoseOrDie(view);
        const Vec2 pt = obs_it.second.x;
        bearing.emplace_back((*cam)(cam->get_ud_pixel(pt)));
        poses.emplace_back(pose.asMatrix());
      }

      const Eigen::Map<const Mat3X> bearing_matrix(bearing[0].data(), 3, bearing.size());
      
      Vec4 Xhomogeneous;
      if (!TriangulateNViewAlgebraic(bearing_matrix, poses, &Xhomogeneous))
      {
        std::cout << "Invalid triangulation.\n\n";
        return;
      }

      const Vec3 X = Xhomogeneous.hnormalized();
      // Test validity of the hypothesis (front of the cameras):
      bool bCheirality = true;
      int i(0);
      double reprojection_error_sum(0.0);
      for (const auto & obs_it : obs)
      {
        const View * view = sfm_data.views.at(obs_it.first).get();
        if (!sfm_data.IsPoseAndIntrinsicDefined(view))
          continue;

        const Pose3 pose = sfm_data.GetPoseOrDie(view);
        bCheirality &= CheiralityTest(bearing[i], pose, X);
        const openMVG::cameras::IntrinsicBase * cam = sfm_data.GetIntrinsics().at(view->id_intrinsic).get();
        const Vec2 pt = obs_it.second.x;
        const Vec2 residual = cam->residual(pose(X), pt);
        reprojection_error_sum += residual.norm();
        ++i;
      }
      if (bCheirality) // Keep the point only if it has a positive depth
      {
        map_triangulated[control_point_it.first] = X;
        map_control_points[control_point_it.first] = landmark.X;
        map_triangulation_errors[control_point_it.first] = reprojection_error_sum/(double)bearing.size();
      }
      else
      {
        std::cout << "Control Point cannot be triangulated (not in front of the cameras)" << std::endl;
        return;
      }
    }

    if (map_control_points.size() < 3)
    {
      std::cout << "Insufficient number of triangulated control points.\n\n";
      return;
    }

    // compute the similarity
    {
      // data conversion to appropriate container
      Mat x1(3, map_control_points.size()),
          x2(3, map_control_points.size());
      
      IndexT id_col = 0;
      for (const auto & cp : map_control_points)
      {
        x1.col(id_col) = map_triangulated[cp.first];
        x2.col(id_col) = cp.second;
        ++id_col;
      }

      std::cout
        << "Control points observation triangulations:\n"
        << x1 << std::endl << std::endl
        << "Control points coords:\n"
        << x2 << std::endl << std::endl;

      Vec3 t;
      Mat3 R;
      double S;
      if (openMVG::geometry::FindRTS(x1, x2, &S, &t, &R))
      {
        openMVG::geometry::Refine_RTS(x1,x2,&S,&t,&R);
        std::cout << "Found transform:\n"
          << " scale: " << S << "\n"
          << " rotation:\n" << R << "\n"
          << " translation: "<< t.transpose() << std::endl;

        //--
        // Apply the found transformation as a 3D Similarity transformation matrix // S * R * X + t
        //--

        const openMVG::geometry::Similarity3 sim(geometry::Pose3(R, -R.transpose() * t/S), S);
        openMVG::sfm::ApplySimilarity(sim, sfm_data);

        // Display some statistics:
        std::stringstream os;
        for (Landmarks::const_iterator iterL = sfm_data.control_points.begin();
          iterL != sfm_data.control_points.end(); ++iterL)
        {
          if (iterL->second.obs.size() < 2)
          {
              continue;
          }

          const IndexT CPIndex = iterL->first;
          // If the control point has not been used, continue...
          if (map_triangulation_errors.find(CPIndex) == map_triangulation_errors.end())
            continue;

          os
            << "CP index: " << CPIndex << "\n"
            << "CP triangulation error: " << map_triangulation_errors[CPIndex] << " pixel(s)\n"
            << "CP registration error: "
            << (sim(map_triangulated[CPIndex]) - map_control_points[CPIndex]).norm() << " user unit(s)"<< "\n\n";
        }
        std::cout << os.str();
      }
      else
      {
        std::cout << "Registration failed. Please check your Control Points coordinates.\n\n";
      }
    }

    //---
    // Bundle adjustment with GCP
    //---
    {
      using namespace openMVG::sfm;
      Bundle_Adjustment_Ceres::BA_Ceres_options options;
      Bundle_Adjustment_Ceres bundle_adjustment_obj(options);
      Control_Point_Parameter control_point_opt(20.0, true);
      if (!bundle_adjustment_obj.Adjust(sfm_data,
          Optimize_Options
          (
            cameras::Intrinsic_Parameter_Type::NONE, // Keep intrinsic constant
            Extrinsic_Parameter_Type::ADJUST_ALL, // Adjust camera motion
            Structure_Parameter_Type::ADJUST_ALL, // Adjust structure
            control_point_opt // Use GCP and weight more their observation residuals
            )
          )
        )
      {
      std::cout << "BA with GCP failed." << std::endl;
      }
    }
  }
}

bool readMarkersPositions(const std::string & sFileName, Landmarks & landmarks)
{
    //---
    // - reads markers positions from file
    // - creates landmarks with empty observations
    //---

    std::ifstream in(sFileName);

    if (!in)
    {
        OPENMVG_LOG_ERROR
            << "loadPairs: Impossible to read the specified file: \"" << sFileName << "\".";
        return false;
    }

    std::string sValue;
    std::vector<std::string> vec_str;
    while (std::getline( in, sValue ) )
    {
        vec_str.clear();
        stl::split(sValue, ' ', vec_str); // 48.1000 5.0000 415.0000 0

        const IndexT str_size (vec_str.size());
        if (str_size < 4) // marker_id x y z
        {
            OPENMVG_LOG_ERROR << "Invalid marker position.";
            continue;
        }

        std::vector<std::string> id_str;
        stl::split(vec_str[0], '.', id_str);
        int markerId = std::stoi(id_str[0]) * 10000 + std::stoi(id_str[1]); // marker ID must be the exact Aruco Marker ID of the specified dictionary

        float x = std::stof(vec_str[1]);
        float y = std::stof(vec_str[2]);
        float z = std::stof(vec_str[3]);

        landmarks[markerId].X = Vec3(x, y, z);
    }

    return true;
}

int main(int argc, char **argv)
{
    CmdLine cmd;

    std::string sSfM_Data_Filename;
    std::string sMarkers_Positions_Filename;
    std::string sOutDir = "";

    // required
    cmd.add( make_option('i', sSfM_Data_Filename, "input_file") );
    cmd.add( make_option('m', sMarkers_Positions_Filename, "markers_positions_file") );
    cmd.add( make_option('o', sOutDir, "outdir") );

    try {
        if (argc == 1) throw std::string("Invalid command line parameter.");
        cmd.process(argc, argv);
    } catch (const std::string& s) {
        OPENMVG_LOG_INFO
            << "Usage: " << argv[0] << '\n'
            << "[-i|--input_file] a SfM_Data file \n"
            << "[-o|--outdir path] \n"            
            << "[-m|--markers_positions_file] a text file with id x y z for each marker \n";

        OPENMVG_LOG_ERROR << s;
        return EXIT_FAILURE;
    }

    OPENMVG_LOG_INFO
        << " You called : " << "\n"
        << argv[0] << "\n"
        << "--input_file " << sSfM_Data_Filename << "\n"
        << "--outdir " << sOutDir << "\n"
        << "--markers_positions_file " << sMarkers_Positions_Filename << "\n";

    // Create output dir
    if (!stlplus::folder_exists(sOutDir))
    {
        if (!stlplus::folder_create(sOutDir))
        {
            OPENMVG_LOG_ERROR << "Cannot create output directory";
            return EXIT_FAILURE;
        }
    }

    //---------------------------------------
    // a. Load input scene
    //---------------------------------------
    SfM_Data sfm_data;
    if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(ALL))) {
        OPENMVG_LOG_ERROR
            << "The input file \""<< sSfM_Data_Filename << "\" cannot be read";
        return EXIT_FAILURE;
    }

    auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250);

    cv::Mat image;
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

    //---------------------------------------
    // b. Load markers and detect
    //---------------------------------------
    Landmarks landmarks;
    readMarkersPositions(sMarkers_Positions_Filename, landmarks);

    for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
    {        
        Views::const_iterator iterViews = sfm_data.views.begin();
        std::advance(iterViews, i);
        const View * view = iterViews->second.get();

        if (!sfm_data.IsPoseAndIntrinsicDefined(view))
            continue;

        const std::string sView_filename = stlplus::create_filespec(sfm_data.s_root_path, view->s_Img_path);

        image = cv::imread(sView_filename);
        
        cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds);

        for (int j = 0; j < markerIds.size(); j++)
        {
            int markerId = markerIds[j] * 10000 + 1000;
            
            if (landmarks.find(markerId) != landmarks.end())
            {
              Vec2 firstCorner = Vec2(markerCorners[j][0].x, markerCorners[j][0].y);
              landmarks[markerId].obs[view->id_view] = Observation(firstCorner, 0);

              Vec2 secondCorner = Vec2(markerCorners[j][1].x, markerCorners[j][1].y);
              landmarks[markerId + 1000].obs[view->id_view] = Observation(secondCorner, 0);

              Vec2 thirdCorner = Vec2(markerCorners[j][2].x, markerCorners[j][2].y);
              landmarks[markerId + 2000].obs[view->id_view] = Observation(thirdCorner, 0);

              Vec2 fourthCorner = Vec2(markerCorners[j][3].x, markerCorners[j][3].y);
              landmarks[markerId + 3000].obs[view->id_view] = Observation(fourthCorner, 0);
            }
        }
    }

    sfm_data.control_points = landmarks;    

    Save(sfm_data,
       stlplus::create_filespec(sOutDir, "sfm_data_cp", ".json"),
       ESfM_Data(ALL));

    //---------------------------------------
    // c. GCP registration
    //---------------------------------------
    gcpRegister(sfm_data);

    //-- Export scene with GCP to disk
    OPENMVG_LOG_INFO << "...Export SfM_Data to disk.";
    Save(sfm_data,
       stlplus::create_filespec(sOutDir, "sfm_data_reg", ".ply"),
       ESfM_Data(ALL));

    //-- Export scene extrinsics to disk
    OPENMVG_LOG_INFO << "...Export SfM_Data to disk (extrinsics only).";
    Save(sfm_data,
       stlplus::create_filespec(sOutDir, "sfm_data_camera_poses", ".json"),
       ESfM_Data(EXTRINSICS));

    //-------------------------------------------------------------------------
    // d. Triangulate position of unknown markers and export them in a txt file
    //-------------------------------------------------------------------------
    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250);

    markerIds.clear();
    markerCorners.clear();

    Hash_Map<IndexT, Observations> markersObservations;

    for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
    {        
        Views::const_iterator iterViews = sfm_data.views.begin();
        std::advance(iterViews, i);
        const View * view = iterViews->second.get();

        if (!sfm_data.IsPoseAndIntrinsicDefined(view))
            continue;

        const std::string sView_filename = stlplus::create_filespec(sfm_data.s_root_path, view->s_Img_path);

        image = cv::imread(sView_filename);
        
        cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds);

        for (int j = 0; j < markerIds.size(); j++)
        {
            Vec2 firstCorner = Vec2(markerCorners[j][0].x, markerCorners[j][0].y);
            markersObservations[markerIds[j]][view->id_view] = Observation(firstCorner, view->id_view);
        }

        cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);

        cv::imwrite(stlplus::create_filespec(sOutDir, view->s_Img_path), image); // just for Debug
    }

    SfM_Data_Structure_Computation_Robust triangulation(2.0); // 2.0 pixels of max reproj error

    Hash_Map<IndexT, Vec3> markers;

    for (auto const& x : markersObservations)
    {
        auto observations = x.second;

        Landmark landmark;

        bool result = triangulation.robust_triangulation(sfm_data, observations, landmark); 

        if (result)
        {
          markers[x.first] = Vec3(landmark.X.x(), landmark.X.y(), landmark.X.z());
        }
    }

    std::string sFileName(stlplus::create_filespec(sOutDir, "detected_markers", ".txt"));

    std::ofstream outputFile(sFileName);
    
    if (!outputFile)
    {
        OPENMVG_LOG_ERROR
            << "loadPairs: Impossible to read the specified file: \"" << sFileName << "\".";
        return false;
    }

    for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
    {
      if (markers.find(i) != markers.end())
      {
        outputFile << i << " " << markers[i].x() << " " << markers[i].y() << " " << markers[i].z() << std::endl;
      }
    }
}