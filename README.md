Unofficial implementation of Lan-grasp (under development)

# Lan-grasp: Using Large Language Models  for Semantic Object Grasping

## Installation

### Client

```bash
pip install -r requirements.txt
```

### Server

Install the basic requirements:

```bash
pip install -r requirements.txt
```

Select preferences and run the command to install [PyTorch](https://pytorch.org/get-started/previous-versions/) locally.

- [Contact-GraspNet](https://github.com/elchun/contact_graspnet_pytorch)

```bash
pip install provider pyrender
```

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO.git)

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)

```bash
pip install accelerate huggingface_hub[hf_xet] qwen-vl-utils[decord] transformers==4.50.3
```

- [Segment Anything V2](https://github.com/facebookresearch/sam2)

```bash
pip install hydra-core iopath
```

## Starup

### Client

### Server

Start the API server

```bash
uvicorn server:app
```
