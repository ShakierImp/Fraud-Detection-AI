# FraudGuardian AI — Troubleshooting FAQ

This FAQ helps resolve common setup and execution issues when running the FraudGuardian AI project.  
Solutions are platform-aware (Windows, macOS, Linux, Docker). Each entry provides **Problem → Solution → Quick Fix → Prevention**.

---

## 1. Kaggle API Credentials Not Set Up

**Problem:**  
Dataset download fails with:  
403 - Forbidden: You must download Kaggle dataset with API credentials

markdown
Copy code

**Solution:**  
1. Create API token from your Kaggle account:  
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)  
   - Click **"Create New API Token"** → downloads `kaggle.json`  
2. Place it in the correct location:  
   - Windows: `C:\Users\<YourUser>\.kaggle\kaggle.json`  
   - Linux/macOS: `~/.kaggle/kaggle.json`  
3. Set permissions (Linux/macOS only):  
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
Quick Fix:

bash
Copy code
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
Prevention:
Always back up your kaggle.json and keep it private.

2. Port 8000 or 8501 Already in Use
Problem:
API/Streamlit fails with error:

makefile
Copy code
OSError: [Errno 98] Address already in use
Solution:
Kill the process using the port.

Linux/macOS:

bash
Copy code
lsof -i:8000
kill -9 <PID>
Windows (PowerShell):

powershell
Copy code
netstat -ano | findstr :8000
taskkill /PID <PID> /F
Quick Fix:
Linux/macOS:

bash
Copy code
kill -9 $(lsof -t -i:8000)
Prevention:
Stop services properly with CTRL+C or docker-compose down.

3. Missing Dependencies / ModuleNotFoundError
Problem:
Running scripts throws:

vbnet
Copy code
ModuleNotFoundError: No module named 'xyz'
Solution:
Reinstall requirements:

bash
Copy code
pip install -r requirements.txt
Quick Fix:

bash
Copy code
pip install <missing-package>
Prevention:
Always activate the virtual environment before running (venv\Scripts\activate or source venv/bin/activate).

4. Docker Permission Issues
Problem:
Error when running Docker:

pgsql
Copy code
permission denied while trying to connect to the Docker daemon socket
Solution:

Linux/macOS: Add user to Docker group:

bash
Copy code
sudo usermod -aG docker $USER
newgrp docker
Windows: Run Docker Desktop as Administrator.

Quick Fix:

bash
Copy code
sudo docker run hello-world
Prevention:
Always log out and back in after modifying groups.

5. Virtual Environment Activation Problems
Problem:
Cannot activate venv in PowerShell (Windows):

csharp
Copy code
execution of scripts is disabled on this system
Solution:
Allow scripts:

powershell
Copy code
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
Then activate:

powershell
Copy code
.\venv\Scripts\activate
Quick Fix:

powershell
Copy code
Set-ExecutionPolicy Bypass -Scope Process
Prevention:
Use PowerShell as Administrator and set policy once.

6. CUDA / GPU Issues with TensorFlow
Problem:
TensorFlow errors like:

csharp
Copy code
Could not load dynamic library 'cudart64_110.dll'
Solution:

Verify GPU support:

bash
Copy code
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
If empty → reinstall correct version:

bash
Copy code
pip install tensorflow==2.12.0
Quick Fix:
Run on CPU:

bash
Copy code
set CUDA_VISIBLE_DEVICES=""    # Windows (CMD)
export CUDA_VISIBLE_DEVICES="" # Linux/macOS
Prevention:
Install GPU drivers + CUDA toolkit matching TensorFlow version.

7. Memory Errors During Training
Problem:
Process killed or error:

bash
Copy code
Killed: 9  (Linux/macOS)
MemoryError (Windows)
Solution:

Use small dataset:

bash
Copy code
python src/scripts/train_model.py --fast
Increase swap (Linux):

bash
Copy code
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
Quick Fix:
Run with --fast flag.

Prevention:
Avoid training with full dataset on low-RAM machines. Use Google Colab if needed.

8. File Path / Permission Issues
Problem:
File not found or permission denied errors:

vbnet
Copy code
FileNotFoundError: 'data/transactions.csv'
PermissionError: [Errno 13] Permission denied
Solution:

Ensure dataset exists:

bash
Copy code
ls data/
Fix permissions (Linux/macOS):

bash
Copy code
chmod -R 755 data/
On Windows, right-click folder → Properties → Security → Give Full Control.

Quick Fix:

bash
Copy code
chmod -R 777 data/
Prevention:
Always run scripts from project root. Keep datasets inside data/.

✅ Quick Recap Commands
Kill API process:

bash
Copy code
kill -9 $(lsof -t -i:8000)
Reactivate venv:

bash
Copy code
source venv/bin/activate
Test API health:

bash
Copy code
curl http://localhost:8000/health
End of Troubleshooting Guide 🚑

yaml
Copy code

---

⚡ This gives you a **ready-to-use FAQ troubleshooting doc** with 8+ real issues covered, copy-pasteable fixes, platform-specific instructions, and prevention tips.  

Want me to also make a **short cheatsheet version** (1-page quick fixes only) for judges, alongside this full doc?