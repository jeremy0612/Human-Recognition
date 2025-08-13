# Human-Recognition

Simple Human Detection in crowd places without tracker targeting for using as a function of Robot - Guess Greeting and currently facing some practical issue.

## Performance Optimization

### Memory Management

The system automatically:
- Sets GPU memory fraction to 80% to prevent OOM errors
- Clears GPU cache every 100 frames
- Falls back to CPU if GPU inference fails

### Model Optimization

For better performance, consider:
- Using a smaller model (e.g., `yolov8n.pt` for speed, `yolov8s.pt` for accuracy)
- Adjusting batch size based on your GPU memory
- Using TensorRT optimization (advanced)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce memory fraction in `webrtc_server.py`
   - Use smaller YOLO model
   - Reduce input resolution

2. **PyTorch not using GPU**
   - Ensure PyTorch was installed with CUDA support
   - Check `torch.cuda.is_available()`
   - Verify CUDA version compatibility

3. **Slow performance**
   - Check GPU utilization with `nvidia-smi`
   - Ensure model is loaded on GPU with `model.to('cuda')`
   - Monitor memory usage

### Debug Commands

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check PyTorch CUDA info
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# Test GPU computation
python -c "import torch; x = torch.randn(1000, 1000).cuda(); print(torch.mm(x, x).shape)"
```
