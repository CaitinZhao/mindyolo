#include <sys/time.h>

#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "common_inc/infer.h"

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(data_path, ".", "data path");
DEFINE_int32(image_size, 640, "image size");
DEFINE_string(fusion_switch_path, "../fusion_switch.cfg", "fusion switch path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_type, "CPU", "device type");

int resize_pad_op(const MSTensor &input, MSTensor *output) {
  std::vector<int64_t> shape = input.Shape();
  auto r = FLAGS_image_size / max(shape[0], shape[1]);
  auto w = shape[1] * r;
  auto h = shape[0] * r;
  bool do_resize = false;
  bool do_pad = false;
  if (r != 1) {
    auto interpolation = InterpolationMode::kLinear;
    if (r < 1) {
      interpolation = InterpolationMode::kArea;
    }
    std::shared_ptr<TensorTransform> resize(new Resize({h, w}, interpolation));
    do_resize = true;
  }
  if (FLAGS_image_size != shape[0] || FLAGS_image_size != shape[1])) {
    auto dh = (FLAGS_image_size - h) / 2;
    auto dw = (FLAGS_image_size - w) / 2;
    auto top = (int)round(dh - 0.1);
    auto bottom = (int)round(dh + 0.1);
    auto left = (int)round(dw - 0.1);
    auto right = (int)round(dw + 0.1);
    std::shared_ptr<TensorTransform> pad(new Pad({left, top, right,bottom}, {114, 114, 114}));
    do_pad = true;
  }
  auto ret = kSuccess;
  if (do_resize && do_pad) {
    Execute resize_pad({resize, pad});
    ret = resize_pad(input, output);
  } else if (do_resize) {
    Execute resize_op({resize});
    ret = resize_op(input, output);
  } else if (do_pad) {
    Execute pad_op({pad});
    ret = pad_op(input, output);
  }
  if (ret != kSuccess) {
    std::cout << "ERROR: resize and pad failed." << std::endl;
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_fusion_switch_path).empty()) {
    std::cout << "Invalid fusion switch path" << std::endl;
    return 1;
  }

  auto ascend = std::make_shared<mindspore::AscendDeviceInfo>();
  ascend->SetDeviceID(FLAGS_device_id);
  if (!FLAGS_fusion_switch_path.empty()) {
    ascend->SetFusionSwitchConfigPath(FLAGS_fusion_switch_path);
  }
  Status ret;
  Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, ascend, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }

  std::vector<MSTensor> model_inputs = model.GetInputs();
  if (model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }

    auto all_files = GetAllFiles(FLAGS_dataset_path);
  if (all_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }

  std::map<double, double> costTime_map;
  size_t size = all_files.size();

  std::shared_ptr<TensorTransform> decode(new Decode());
  std::shared_ptr<TensorTransform> hwc2chw(new HWC2CHW());
  std::shared_ptr<TensorTransform> normalize(new Normalize({0.0, 0.0, 0.0}, {255.0, 255.0, 255.0}));

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << all_files[i] << std::endl;

    auto image = ReadFileToTensor(all_files[i]);
    // decode
    auto decode_img = MSTensor();
    Execute composeDecode({decode});
    composeDecode(image, &decode_img);
    // resize and pad
    auto resize_img = MSTensor();
    resize_pad_op(decode_img, &resize_img);
    // normalize
    auto img = MSTensor();
    Execute composeNormalize({normalize, hwc2chw});
    composeDecode(resize_img, &img);

    std::vector<MSTensor> model_inputs = model.GetInputs();
    if (model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
    }
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(), img.Data().get(),
                        img.DataSize());

    gettimeofday(&start, nullptr);
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << all_files[i] << " failed." << std::endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    WriteResult(all_files[i], outputs);
  }
  double average = 0.0;
  int inferCount = 0;

  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = iter->second - iter->first;
    average += diff;
    inferCount++;
  }
  average = average / inferCount;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: " << average << " ms of infer_count " << inferCount << std::endl;
  std::cout << "NN inference cost average time: " << average << "ms of infer_count " << inferCount << std::endl;
  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}
