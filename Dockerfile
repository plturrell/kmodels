# Use the official PyTorch image as a base
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /workspace

# Copy the entire project to the working directory
COPY . .

# Install Python dependencies
# We can add a RUN command here if you have a requirements.txt file

# Set a default command (optional)
CMD ["/bin/bash"]
