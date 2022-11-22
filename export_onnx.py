import os
import torch
import cv2
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import models
from build_utils import img_utils

device = torch.device("cpu")
models.ONNX_EXPORT = True


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    img_size = 512  
    cfg = "cfg/yolov3-spp.cfg"
    weights = "weights/yolov3-spp-ultralytics-{}.pt".format(img_size)
    assert os.path.exists(cfg), "cfg file does not exist..."
    assert os.path.exists(weights), "weights file does not exist..."

    input_size = (img_size, img_size)  


    model = models.Darknet(cfg, input_size)

    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)
    model.eval()

    img_path = "test.jpg"
    img_o = cv2.imread(img_path)  
    assert img_o is not None, "Image Not Found " + img_path


    img = img_utils.letterbox(img_o, new_shape=input_size, auto=False, color=(0, 0, 0))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  
    img = np.ascontiguousarray(img).astype(np.float32)

    img /= 255.0 
    img = np.expand_dims(img, axis=0) 
    x = torch.tensor(img)
    torch_out = model(x)

    save_path = "yolov3spp.onnx"

    torch.onnx.export(model,                       
                      x,                           
                      save_path,                   
                      export_params=True,         
                      opset_version=12,            
                      do_constant_folding=True,    
                      input_names=["images"],       
                      output_names=["prediction"],
                      dynamic_axes={"images": {0: "batch_size"},  
                                    "prediction": {0: "batch_size"}})



    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)


    ort_session = onnxruntime.InferenceSession(save_path)


    ort_inputs = {"images": to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)


    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
