


import torch

if __name__ == "__main__":
   

    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"we have device cuda:{i} available")
    model0= torch.jit.load("/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_model/4cls_RGB_5_1_2_3_scripted.pt"
    )
    print(f"model without mapping to any device is on {model0.device}")
    model1 = torch.jit.load("/gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/deep_learning_model/4cls_RGB_5_1_2_3_scripted.pt",
        map_location=torch.device("cuda:1")
    )
    model2= model1.to("cuda:1")
    print(f"model without mapping to cuda:1 is on {model1.device}")
    print(f"model without mapping and using .to to cuda:1 is on {model2.device}")


