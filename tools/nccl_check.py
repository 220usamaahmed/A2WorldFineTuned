import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
t = torch.ones(1024, 1024, device="cuda") * (r + 1)
t0 = time.time(); dist.all_reduce(t); torch.cuda.synchronize()
print(f"rank {r}: all_reduce ok, value {t[0,0].item()} (expect 3.0), {time.time()-t0:.2f}s", flush=True)
g = dist.new_group([0, 1]); dist.all_reduce(t, group=g); torch.cuda.synchronize(); print(f"rank {r}: new_group ok", flush=True)
dist.destroy_process_group()
