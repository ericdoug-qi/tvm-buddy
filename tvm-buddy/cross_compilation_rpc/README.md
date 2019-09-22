# [Cross Compilation and RPC](https://docs.tvm.ai/tutorials/cross_compilation_and_rpc.html#tutorial-cross-compilation-and-rpc)


## Build TVM Runtime on Device

```bash
git clone --recursive https://github.com/dmlc/tvm
cd tvm
make runtime -j2
```

After building the runtime successfully, we need to set environment variables in ~/.bashrc file. We can edit ~/.bashrc using vi ~/.bashrc and add the line below (Assuming your TVM directory is in ~/tvm):

```bash
export PYTHONPATH=$PYTHONPATH:~/tvm/python
```

## Set Up RPC Server on Device

```bash
python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
```

## Declare and Cross Compile Kernel on Local Machine



