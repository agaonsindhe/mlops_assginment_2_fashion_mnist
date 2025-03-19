# MLOps Fashion MNIST Project

This project uses **DVC (Data Version Control)** to manage and version large datasets with integration to **Azure Blob Storage**. 

## **Setup Instructions**

### 1. **Clone the Repository**
First, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/mlops-fashion-mnist.git
cd mlops-fashion-mnist

2. Create and Activate a Virtual Environment
It’s recommended to use a virtual environment to keep dependencies isolated. If you're using Python 3.x, follow these steps:

For Linux/macOS:
bash
Copy code
python3 -m venv venv
source venv/bin/activate
For Windows:
bash
Copy code
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
Once the virtual environment is activated, install the necessary dependencies:

bash
Copy code
pip install -r requirements.txt
4. Install DVC with Azure Support
To interact with Azure Blob Storage, ensure dvc-azure is installed:

bash
Copy code
pip install "dvc[azure]"

This will install DVC and the necessary Azure plugin.

5. Set Up DVC for Azure Blob Storage
DVC uses Azure Blob Storage to store large files. You will need to configure it by updating the connection_string.

Get your Azure Connection String:

Navigate to your Azure Storage Account → Access Keys → Copy Key1 or Key2.
Configure DVC Remote: Run the following commands to set the correct connection string for Azure:

bash
Copy code
dvc remote modify azure_remote url azure://fashion-minist
dvc remote modify azure_remote connection_string "<your-connection-string>"
Replace <your-connection-string> with your actual Azure Storage Account Connection String.

6. Run the Project
You can now run the project and start working with the dataset. To download the dataset or push new data to Azure, use the following DVC commands:

To pull the dataset (if it exists in Azure):
bash
Copy code
dvc pull
