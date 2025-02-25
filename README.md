# OpenVINO_genAI_tests
testing the OpenVINO genAI 2025.0 on AI PC


```
pip install openvino-genai==2025.0.0
pip install optimum[openvino,nncf] torchvision evaluate
```

model used
- [SmolLM2-360M-Instruct-openvino-fp16](https://huggingface.co/AIFunOver/SmolLM2-360M-Instruct-openvino-fp16)
- [phi-4-ov](https://huggingface.co/llmware/phi-4-ov)
- [Mistral-Small-24B-Instruct-2501-int4_asym-ov](https://huggingface.co/Echo9Zulu/Mistral-Small-24B-Instruct-2501-int4_asym-ov)
- [Qwen2.5-14B-Instruct-1M-openvino-4bit](https://huggingface.co/AIFunOver/Qwen2.5-14B-Instruct-1M-openvino-4bit)

---

The 24B is quite slow 2 tk/sec, and loading time is very high

---

