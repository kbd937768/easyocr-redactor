# easyocr-redactor
Redacts keywords from images

HOW TO USE  
1.  
Download checker_v3_gpu_easyocr.py  

2.  
Download and install python (3.11 or higher), install torch and torchvision, example for 5060ti:  
>python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132  

and lastly easyocr:   
>python -m pip install --upgrade easyocr opencv-python pillow numpy  

3.  
  - File folder structure should look like this:  
   - Videos  
      - frames  
         - pic_0001.png  
         - pic_0002.png  
         - ...  
      - checker_v3_gpu_easyocr.py  

4.  
To run code:
>python checker_v3_gpu_easyocr.py  

Output should look like this:  
>20755/21625 | frame_020755.png | Total: 2.33s | Read: 0.19s | OCR: 1.02s | Redact: 0.01s | Save: 1.09s | Redacted: 3 | FPS: 0.34 | Elapsed: 17h 7m 44s | ETA: 43m 4s  

**Example pictures:**  
Before   
<img width="3840" height="2160" alt="frame_gaia2_slp_000028" src="https://github.com/user-attachments/assets/6946cc22-0b63-4df5-a8dc-a9d4d0a95b11" />

After   
<img width="3840" height="2160" alt="frame_gaia2_slp_000028" src="https://github.com/user-attachments/assets/8e527131-30c8-4090-8c03-022944706c5e" />

Processing speed of 4K image with 2x amd epyc 7742, 256GB of ram and 5060ti (16GB) is between 2-3 seconds per frame.    
Processing speed of 4K image with 5 1600x, 16GB of ram and 1660ti is between 1.5-2 seconds per frame.    

for multi pc setup use checker_v7_multi_gpu_easyocr_remote_worker.py.   
And with this one you need to select the devices to use and set the global total workers and worker ids for every device you want to use.   

Example with 2 gpus:   
PS D:\AI Videos\xqc\test> python checker_v7_multi_gpu_easyocr_remote_worker.py

Available Devices:
  [cpu] Central Processing Unit   
  [0] NVIDIA GeForce RTX 5060 Ti   
  [1] NVIDIA GeForce RTX 5060 Ti   

Select LOCAL device(s) to use (example: cpu or 0 or cpu,0 or all): 0,1   

Enter TOTAL number of global workers across all machines (e.g., 3): 2   
Enter the 2 global worker ID(s) for this machine (1 to 2, example: 1 or 1,2): 1,2   

=== Startup ===  
Frames directory: frames_upscaled   
Total frames globally: 60064   
Total global workers: 2   
This machine handles worker IDs: [1, 2]   
Resume mode: skipped 0 already-processed frames assigned to this machine.   
Frames to process NOW on this machine: 60064    
  Local Device [0] (Global Worker 1): 30032 frame(s)    
  Local Device [1] (Global Worker 2): 30032 frame(s)    
Processing mode: in-place overwrite    
Press Ctrl+C to stop safely.    


# Other thing
If there are any coders who want a challenge, try to get the old paddle ocr code working.
