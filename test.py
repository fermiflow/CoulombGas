import torch 
torch.set_num_threads(1)
import time 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--param1", type=float, default=2.0, help="param1")
    parser.add_argument("--param2", type=int, default=6, help="param2")
    parser.add_argument("--param3", type=int, default=0, help="param3")
    args = parser.parse_args()
    print("param1 = %f, param2 = %d, param3 = %d." % (args.param1, args.param2, args.param3))

    M, N = 5000, 5000 
    for dtype in [torch.float32, torch.float64]:
        for device in ['cpu', 'cuda']:
            A = torch.randn(M, N, dtype=dtype, device=device)
            start = time.time()
            torch.svd(A)
            print(A.dtype, A.device, time.time()-start)
