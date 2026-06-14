# easyocr-redactor
Redacts keywords from images

HOW TO USE  
1.  
Download checker_v2_gpu_easyocr.py  

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
      - checker_v2_gpu_easyocr.py  

4.  
To run code:
>python checker_v2_gpu_easyocr.py  

Output should look like this:  
>20755/21625 | frame_020755.png | Total: 2.33s | Read: 0.19s | OCR: 1.02s | Redact: 0.01s | Save: 1.09s | Redacted: 3 | FPS: 0.34 | Elapsed: 17h 7m 44s | ETA: 43m 4s  

**Example pictures:**  
Before   
<img width="3840" height="2160" alt="frame_gaia2_slp_000028" src="https://github.com/user-attachments/assets/6946cc22-0b63-4df5-a8dc-a9d4d0a95b11" />

After   
<img width="3840" height="2160" alt="frame_gaia2_slp_000028" src="https://github.com/user-attachments/assets/8e527131-30c8-4090-8c03-022944706c5e" />

Processing speed of 4K image with 2x amd epyc 7742, 256GB of ram and 5060ti (16GB) is between 2-3 seconds per frame

# Other thing
