import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import numpy as np
from modules.diffusion import Diffusion
from modules.flow import Flow
import os
import mlflow
import imageio
from datetime import datetime
import logging


# Set up logging
def setup_logging(output_dir):
    """Set up logging with file and console handlers, creating output directory if needed"""
  
    # Configure logging
    log_file = os.path.join(output_dir, 'training.log')
    
    # Create a custom formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to create log file handler: {e}")
    
    # Console handler with custom stream to avoid tqdm interference
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)
    
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress MLflow system metrics warnings
    logging.getLogger('mlflow.system_metrics.metrics').setLevel(logging.ERROR)

def plot(X, img_shape, filename):
    X = X.to("cpu")
    X = X.detach().numpy()
    fig = plt.figure(figsize=(5, 5))
    for i in range(25): 
        sub = fig.add_subplot(5, 5, i+1)
        img = X[i, :].reshape(img_shape)
        if len(img_shape) == 3:
            img = np.moveaxis(img, 0, -1)
        sub.imshow(img, cmap=plt.cm.gray)
    
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_denoising_gif(noisy_images, denoised_images, filename, fps=10):
    """Create a GIF showing the denoising process"""
    frames = []
    for i in range(len(noisy_images)):
        # Convert to numpy and normalize to [0, 255]
        frame = noisy_images[i].cpu().detach().numpy()
        # Handle different input shapes
        if len(frame.shape) == 4:  # [batch, channel, height, width]
            # Take first image from batch
            frame = frame[0]
        
        # Handle channel dimension
        if len(frame.shape) == 3:
            if frame.shape[0] == 1:  # Grayscale [1, height, width]
                frame = frame.squeeze(0)
            else:  # RGB [3, height, width]
                frame = np.moveaxis(frame, 0, -1)  # Move channels to last dimension
        
        # Normalize to [0, 255]
        frame = ((frame + 1) / 2 * 255).astype(np.uint8)
        frames.append(frame)
    
    # Add final denoised result
    final = denoised_images[-1].cpu().detach().numpy()
    if len(final.shape) == 4:  # [batch, channel, height, width]
        final = final[0]
    
    # Handle channel dimension for final image
    if len(final.shape) == 3:
        if final.shape[0] == 1:  # Grayscale [1, height, width]
            final = final.squeeze(0)
        else:  # RGB [3, height, width]
            final = np.moveaxis(final, 0, -1)  # Move channels to last dimension
    
    final = ((final + 1) / 2 * 255).astype(np.uint8)
    frames.append(final)
    
    # Save as GIF
    imageio.mimsave(filename, frames, fps=fps)

def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])  
        train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                        transform=transform)
        img_shape = (28, 28)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                        transform=transform)  
        img_shape = (3, 32, 32)
    
    return train_dataset, img_shape

class Trainer:
    def __init__(self, model, dataset_name='mnist',
                  checkpt="./output/chk_rbm.pt", n_epochs=10, 
                  lr=0.01, batch_size=64, device="cpu"): 
        self.model = model
        self.dataset_name = dataset_name
        self.device = device
        self.checkpt = checkpt
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.checkpt_epoch = 0
        self.total_epochs = n_epochs  # Store total epochs separately
        self.best_loss = float('inf')
        self.setup()
        
        # Create output directories
        self.output_dir = os.path.dirname(checkpt)
        self.visualization_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Initialize MLflow tracking
        self.start_mlflow_run()
        
        # Track best checkpoints
        self.max_checkpoints = 10
        self.checkpoint_history = []  # List of (loss, epoch, path) tuples
    
    def start_mlflow_run(self):
        """Initialize MLflow run with model and training parameters"""
        # End any active runs first
        try:
            mlflow.end_run()
        except:
            pass
            
        # Enable system metrics logging
        try:
            mlflow.enable_system_metrics_logging()
            logging.info("Enabled MLflow system metrics logging")
        except Exception as e:
            logging.warning(f"Failed to enable MLflow system metrics logging: {e}")           
        
        # Start new run or resume existing run
        if os.path.exists(self.checkpt):
            try:
                # Try loading checkpoint first
                checkpoint = torch.load(self.checkpt, weights_only=False)
                if 'mlflow_run_id' in checkpoint:
                    # Resume existing run
                    self.run_id = checkpoint['mlflow_run_id']
                    run = mlflow.start_run(run_id=self.run_id)
                    logging.info(f"Resuming MLflow run {self.run_id}")
                    return  # Don't log parameters again for resumed run
                else:
                    # Start new run
                    run = mlflow.start_run()
                    self.run_id = run.info.run_id
                    logging.info(f"Starting new MLflow run {self.run_id}")
            except Exception as e:
                logging.warning(f"Failed to load checkpoint for MLflow run ID: {e}")
                # Start new run if checkpoint loading fails
                run = mlflow.start_run()
                self.run_id = run.info.run_id
                logging.info(f"Starting new MLflow run {self.run_id}")
        else:
            # Start new run
            run = mlflow.start_run()
            self.run_id = run.info.run_id
            logging.info(f"Starting new MLflow run {self.run_id}")
        
        # Only log parameters for new runs
        mlflow.log_params({
            "model_type": self.model.__class__.__name__,
            "dataset": self.dataset_name,
            "n_epochs": self.total_epochs,  # Log total epochs instead of n_epochs
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "device": self.device,
            "checkpoint_path": self.checkpt,
            "model_params": {
                "n_channels": self.model.n_channels if hasattr(self.model, 'n_channels') else None,
                "t_emb_dim": self.model.t_emb_dim if hasattr(self.model, 't_emb_dim') else None,
                "n_visible": self.model.n_visible if hasattr(self.model, 'n_visible') else None,
            }
        })

    def end_mlflow_run(self):
        """End the current MLflow run"""
        try:
            mlflow.end_run()
        except:
            pass

    def __del__(self):
        """Cleanup when the trainer is deleted"""
        self.end_mlflow_run()

    def log_metrics(self, metrics, step):
        """Log metrics to MLflow with step"""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logging.warning(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, artifact_path, artifact_name=None):
        """Log artifact to MLflow"""
        try:
            mlflow.log_artifact(artifact_path, artifact_name)
        except Exception as e:
            logging.warning(f"Failed to log artifact to MLflow: {e}")

    def setup(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.set_device("cuda:0")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        
        if os.path.isfile(self.checkpt):
            try:
                # Try loading with weights_only=False first
                checkpoint = torch.load(self.checkpt, weights_only=False)
            except Exception as e:
                logging.warning(f"Failed to load checkpoint with weights_only=False: {e}")
                try:
                    # If that fails, try loading with weights_only=True
                    checkpoint = torch.load(self.checkpt, weights_only=True)
                except Exception as e:
                    logging.error(f"Failed to load checkpoint: {e}")
                    checkpoint = None
            
            if checkpoint is not None:
                try:
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.checkpt_epoch = checkpoint['epoch']
                    self.lr = checkpoint['lr']
                    if 'best_loss' in checkpoint:
                        self.best_loss = checkpoint['best_loss']
                    if 'total_epochs' in checkpoint:
                        # Add new epochs to the total from checkpoint
                        self.total_epochs = checkpoint['total_epochs'] + self.n_epochs
                    else:
                        # If no total_epochs in checkpoint, start from checkpoint epoch
                        self.total_epochs = self.checkpt_epoch + self.n_epochs
                    logging.info(f"Loaded checkpoint from epoch {self.checkpt_epoch}, training for {self.n_epochs} more epochs. Total epochs will be {self.total_epochs}")
                except Exception as e:
                    logging.error(f"Failed to load model state from checkpoint: {e}")
                    self.checkpt_epoch = 0
                    self.best_loss = float('inf')
                    self.total_epochs = self.n_epochs
        else:
            logging.info("No checkpoint found, starting from scratch")
            self.checkpt_epoch = 0
            self.best_loss = float('inf')
            self.total_epochs = self.n_epochs
        
        self.model.to(self.device)
        self.model.train()

    def save_chekpoint(self, epoch_, loss_):
        checkpoint = {
            'epoch': epoch_,
            'state_dict': self.model.state_dict(),
            'lr': self.lr,
            'best_loss': self.best_loss,
            'mlflow_run_id': self.run_id,  # Save MLflow run ID for resumption
            'total_epochs': self.total_epochs  # Save total epochs
        }
        try:
            # Save checkpoint locally
            torch.save(checkpoint, self.checkpt)
            
            # Store old checkpoints before updating history
            old_checkpoints = set(self.checkpoint_history)
            
            # Update checkpoint history
            self.checkpoint_history.append((loss_, epoch_, self.checkpt))
            
            # Sort by loss (best first) and keep only top max_checkpoints
            self.checkpoint_history.sort(key=lambda x: x[0])
            self.checkpoint_history = self.checkpoint_history[:self.max_checkpoints]
            
            # Get new set of checkpoints
            new_checkpoints = set(self.checkpoint_history)
            
            # Find checkpoints that were removed
            removed_checkpoints = old_checkpoints - new_checkpoints
            
            # Delete removed checkpoints from MLflow
            if removed_checkpoints:
                try:
                    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
                    repository = get_artifact_repository(mlflow.get_artifact_uri())
                    for _, epoch, _ in removed_checkpoints:
                        filename = f"checkpoint_epoch_{epoch}"
                        repository.delete_artifacts(filename)
                        logging.info(f"Deleted old checkpoint artifact: {filename}")
                except Exception as e:
                    logging.warning(f"Failed to delete old checkpoint artifacts from MLflow: {e}")
            
            # Only log the new checkpoint if it's among the best 10
            if (loss_, epoch_, self.checkpt) in self.checkpoint_history:
                mlflow.log_artifact(self.checkpt, f"checkpoint_epoch_{epoch_}")
            
            # Log current loss
            mlflow.log_metric("loss", loss_, step=epoch_)
            
        except Exception as e:
            logging.warning(f"Failed to save checkpoint or log to MLflow: {e}")
        logging.info(f'Saving new checkpt at epoch {epoch_} - Loss: {loss_:.4f}')

    def test_diffusion(self, epoch, diffusion):
        """Test denoising process and create visualizations"""
        self.model.eval()
        with torch.no_grad():
            # Generate sample batch
            if self.dataset_name == 'mnist':
                x = torch.randn((25, 1, 28, 28))
            else:  # cifar10
                x = torch.randn((25, 3, 32, 32))
            x = x.to(self.device)
            
            # Store intermediate results
            noisy_images = []
            denoised_images = []
            
            # Denoising process
            for t in tqdm(reversed(range(diffusion.timesteps)), desc="Denoising test"):
                t_batch = torch.full((x.shape[0],), t, device=self.device)
                noise_pred = self.model(x, t_batch)
                x, _ = diffusion.denoise(x, noise_pred, t_batch)
                
                # Store intermediate results every 100 steps
                if t % 100 == 0:
                    noisy_images.append(x.clone())
                    denoised_images.append(x.clone())
            
            # Create visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save final denoised images
            denoised_path = os.path.join(self.visualization_dir, f'denoised_epoch_{epoch}_{timestamp}.png')
            plot(x, (1, 28, 28) if self.dataset_name == 'mnist' else (3, 32, 32), denoised_path)
            
            # Create and save denoising process GIF
            gif_path = os.path.join(self.visualization_dir, f'denoising_process_epoch_{epoch}_{timestamp}.gif')
            create_denoising_gif(noisy_images, denoised_images, gif_path)
            
            # Log artifacts under visualizations directory
            self.log_artifact(denoised_path, "visualizations")
            self.log_artifact(gif_path, "visualizations")
            
            # Calculate and log image statistics
            with torch.no_grad():
                mean_value = x.mean().item()
                std_value = x.std().item()
                self.log_metrics({
                    "denoised_mean": mean_value,
                    "denoised_std": std_value,
                }, step=epoch)
            
        self.model.train()

    def train_diffusion(self, train_loader):    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        diffusion = Diffusion(timesteps=1000, device=self.device)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        last_lr = self.lr  # Track last learning rate to detect changes
        
        # Resume from checkpoint
        start_epoch = self.checkpt_epoch
        end_epoch = start_epoch + self.n_epochs 
        logging.info(f"Starting training from epoch {start_epoch + 1}")
        
        for epoch in range(start_epoch, end_epoch):
            epoch_loss = []
            epoch_grad_norm = []
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.n_epochs}')
            
            for batch_idx, (data, _) in enumerate(progress_bar):
                data = data.to(self.device)
                if self.dataset_name == 'mnist':
                    # Ensure correct shape for MNIST: [batch_size, 1, 28, 28]
                    if len(data.shape) == 3:
                        data = data.unsqueeze(1)
                    data = data * 2 - 1  # Normalize to [-1, 1]
                else:  # cifar10
                    data = data * 2 - 1  # Normalize to [-1, 1]
                
                t = torch.randint(0, diffusion.timesteps, (data.shape[0],))
                t = t.to(self.device)
                noise = torch.randn_like(data).to(self.device)
                noisy_data = diffusion.add_noise(data, noise, t)
                
                optimizer.zero_grad()
                pred_noise = self.model(noisy_data, t)
                loss = loss_fn(pred_noise, noise)
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                epoch_grad_norm.append(grad_norm.item())
                
                optimizer.step()
                
                epoch_loss.append(loss.item())
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log batch metrics every 100 batches
                if batch_idx % 100 == 0:
                    self.log_metrics({
                        "batch_loss": loss.item(),
                        "batch_grad_norm": grad_norm.item(),
                        "batch_noise_pred_mean": pred_noise.mean().item(),
                        "batch_noise_pred_std": pred_noise.std().item(),
                    }, step=epoch * len(train_loader) + batch_idx)
            
            mean_loss = np.mean(epoch_loss)
            mean_grad_norm = np.mean(epoch_grad_norm)
            epoch_ = epoch + 1
            
            # Update learning rate based on loss
            scheduler.step(mean_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Log if learning rate changed
            if current_lr != last_lr:
                logging.info(f'Learning rate decreased from {last_lr:.6f} to {current_lr:.6f}')
                last_lr = current_lr
            
            # Save checkpoint if best loss
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.save_chekpoint(epoch_, mean_loss)
            
            # Log epoch metrics
            self.log_metrics({
                "epoch_loss": mean_loss,
                "epoch_grad_norm": mean_grad_norm,
                "learning_rate": current_lr,
                "best_loss": self.best_loss,
            }, step=epoch_)
            
            logging.info(f'Epoch {epoch_} - Mean Loss: {mean_loss:.4f} - LR: {current_lr:.6f}')
            
            # Run denoising test after each epoch
            self.test_diffusion(epoch_, diffusion)            
        return self.model
    
    def test_rbm(self, epoch):
        """Test denoising process and create visualizations"""
        self.model.eval()
        if self.dataset_name == 'mnist':
                img_shape = (28, 28)
        else:  # cifar10
            img_shape = (3, 32, 32)
        with torch.no_grad():
            # Generate sample batch
            x = torch.randn((25, self.model.n_visible))
            x = x.to(self.device)    

            # Denoising process
            v_gen = self.model.generate(x)
            w0 = self.model.rbm_modules[0].weight
            
            # Create visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save final denoised images
            denoised_path = os.path.join(self.visualization_dir, f'denoised_epoch_{epoch}_{timestamp}.png')
            filter_path = os.path.join(self.visualization_dir, f'filter_epoch_{epoch}_{timestamp}.png')

             # plot the results
            plot(w0, img_shape, filter_path)
            plot(v_gen, img_shape, denoised_path)
            
            # Log artifacts under visualizations directory
            self.log_artifact(denoised_path, "visualizations")
            self.log_artifact(filter_path, "visualizations")
            
            # Calculate and log image statistics
            with torch.no_grad():
                mean_value = x.mean().item()
                std_value = x.std().item()
                self.log_metrics({
                    "denoised_mean": mean_value,
                    "denoised_std": std_value,
                }, step=epoch)
            
        self.model.train()

    def train_rbm(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        last_lr = self.lr  # Track last learning rate to detect changes
         # Resume from checkpoint
        start_epoch = self.checkpt_epoch
        end_epoch = start_epoch + self.n_epochs 
        logging.info(f"Starting training from epoch {start_epoch + 1}")
        for epoch in range(start_epoch, end_epoch):
            optimizer.zero_grad()
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_loss = []
            epoch_grad_norm = []
            for _, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                input = data.view(-1, self.model.n_visible)
                loss = self.model.fit(input, current_lr, self.batch_size)
                epoch_loss.append(loss.item())
                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                epoch_grad_norm.append(grad_norm.item())
                optimizer.step()
        
            mean_loss = np.mean(epoch_loss)
            mean_grad_norm = np.mean(epoch_grad_norm)
            epoch_ = epoch + 1
            
            # Update learning rate based on loss
            scheduler.step(mean_loss)            
            
            # Log if learning rate changed
            if current_lr != last_lr:
                logging.info(f'Learning rate decreased from {last_lr:.6f} to {current_lr:.6f}')
                last_lr = current_lr
            
            # Save checkpoint if best loss
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.save_chekpoint(epoch_, mean_loss)
            
            # Log epoch metrics
            self.log_metrics({
                "epoch_loss": mean_loss,
                "epoch_grad_norm": mean_grad_norm,
                "learning_rate": current_lr,
                "best_loss": self.best_loss,
            }, step=epoch_)
            
            logging.info(f'Epoch {epoch_} - Mean Loss: {mean_loss:.4f} - LR: {current_lr:.6f}')
            if epoch % 10 == 0:
                self.test_rbm(epoch)
            
        return self.model

    # flow matching
    def train_fm(self, train_loader):    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        flow = Flow(device=self.device)

        for epoch in range(self.n_epochs):
            loss_ = []
            for _, (data, _) in enumerate(tqdm(train_loader)):
                data = data.to(self.device)
                if self.dataset_name == 'mnist':
                    # [batch_size, 1, 28, 28]
                    data.unsqueeze(3)
                    data = data * 2 - 1
                x0 = torch.randn_like(data).to(self.device)
                t = torch.rand(data.shape[0]).to(self.device)
                xt, target = flow.sample_xt(data, x0, t)
                optimizer.zero_grad()
                pred = self.model(xt, t)
                loss = loss_fn(pred, target)
                loss_.append(loss.item())
                loss.backward()
                optimizer.step()

            epoch_ = epoch + self.checkpt_epoch + 1
            self.save_chekpoint(epoch_, np.mean(loss_)) 
        return self.model