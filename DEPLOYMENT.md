# Vesta Deployment Guide - Digital Ocean Droplet

Complete guide for deploying Vesta on a Digital Ocean droplet using Docker.

**Why Digital Ocean?**
- ✅ Persistent storage for SQLite database
- ✅ Predictable performance (always-on)
- ✅ Low cost ($4-6/month)
- ✅ Full control over the environment

---

## Prerequisites

- Digital Ocean account
- Domain name (optional, only for HTTPS)
- SSH key configured
- Basic knowledge of Linux command line

---

## 1. Create and Configure Droplet

### Create Droplet

1. Log into [Digital Ocean](https://cloud.digitalocean.com)
2. Click "Create" → "Droplets"
3. Choose configuration:
   - **Distribution**: Ubuntu 22.04 LTS
   - **Plan**: Basic Shared CPU
     - **$4/month**: 512MB RAM, 1 vCPU, 10GB SSD (sufficient for low traffic)
     - **$6/month**: 1GB RAM, 1 vCPU, 25GB SSD (recommended)
   - **Region**: Choose closest to your users
   - **Authentication**: SSH key (recommended over password)
   - **Hostname**: `vesta` (or your preferred name)

4. Click "Create Droplet"

### Initial Server Setup

SSH into your droplet as root:
```bash
ssh root@your_droplet_ip
```

Update system packages:
```bash
apt update && apt upgrade -y
```

Create a non-root user (recommended):
```bash
adduser vesta
usermod -aG sudo vesta
su - vesta
```

---

## 2. Install Dependencies

### Install Docker

```bash
# Download and run Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (replace 'vesta' with your username)
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and back in for group changes to take effect
exit
```

SSH back in and verify installation:
```bash
docker --version
docker-compose --version
```

Should output something like:
```
Docker version 24.x.x
docker-compose version 1.29.x
```

### Install Git and Python (for model training)

```bash
sudo apt install -y git python3-pip python3-venv
```

---

## 3. Clone Repository

Clone your Vesta repository:

```bash
# Clone the repository (replace with your GitHub username)
git clone https://github.com/yourusername/vesta.git
cd vesta
```

---

## 4. Prepare ML Models

Since your GitHub repo is public, train models directly on the droplet to keep them private.

### Option A: Train on Droplet (Recommended)

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -e packages/vesta_ml

# Train the models
cd packages/vesta_ml
python scripts/preprocess.py  # Creates sample data and preprocesses
python scripts/train.py       # Trains Random Forest models
python scripts/evaluate.py    # Optional: generates metrics and plots

# Verify models were created
ls -lh models/random_forest/
# Should see: mean_model.pkl, lower_model.pkl, upper_model.pkl

ls -lh data/processed/
# Should see: scaler.pkl, cycles_processed.npz

# Return to project root
cd ../..
```

### Option B: Train Locally, Transfer via SCP

If you prefer to train on your local machine:

**On your local machine:**
```bash
# Train models
python packages/vesta_ml/scripts/preprocess.py
python packages/vesta_ml/scripts/train.py

# Package the models
cd packages/vesta_ml
tar czf models.tar.gz models/ data/processed/

# Transfer to droplet (replace user and IP)
scp models.tar.gz vesta@your_droplet_ip:~/vesta/packages/vesta_ml/
```

**On your droplet:**
```bash
# Extract models
cd ~/vesta/packages/vesta_ml
tar xzf models.tar.gz
rm models.tar.gz  # Clean up

# Verify
ls -lh models/random_forest/
ls -lh data/processed/
```

---

## 5. Deploy with Docker Compose

The repository includes a `docker-compose.yml` file for easy deployment.

### Start the Application

```bash
# From the project root
cd ~/vesta

# Build and start the container in detached mode
docker-compose up -d

# This will:
# - Build the Docker image
# - Start the container
# - Mount models and data as read-only volumes
# - Create a persistent volume for the SQLite database
# - Expose the app on port 8000
```

### Verify Deployment

```bash
# Check if container is running
docker-compose ps

# Should show:
# NAME    IMAGE        STATUS   PORTS
# vesta   vesta_vesta  Up       0.0.0.0:8000->8000/tcp

# View logs
docker-compose logs -f

# Test the application
curl http://localhost:8000

# Access from browser
# Visit: http://your_droplet_ip:8000
```

### Seed Initial Data (Optional)

```bash
# Run the seed script inside the container
docker exec -it vesta python seed_cycles.py

# This adds 6 sample cycles so predictions work immediately
```

---

## 6. Docker Compose Commands

Common commands for managing your deployment:

```bash
# View logs (live)
docker-compose logs -f

# View logs (last 100 lines)
docker-compose logs --tail=100

# Restart the application
docker-compose restart

# Stop the application
docker-compose down

# Stop and remove volumes (WARNING: deletes database)
docker-compose down -v

# Rebuild and restart (after code changes)
docker-compose up -d --build

# Check resource usage
docker stats vesta

# Access container shell
docker exec -it vesta bash
```

---

## 7. Configure Firewall

Allow HTTP/HTTPS traffic:

```bash
# Allow port 8000 (HTTP)
sudo ufw allow 8000/tcp

# If using HTTPS (see section 8), allow 443
sudo ufw allow 443/tcp

# Allow SSH (important!)
sudo ufw allow OpenSSH

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

---

## 8. HTTPS with SSL Certificates (Optional)

This section explains how to add HTTPS using `acme.sh` with Digital Ocean DNS and configure Gunicorn to serve SSL directly (no nginx needed).

### Prerequisites for HTTPS

- A domain name (e.g., `vesta.yourdomain.com`)
- Digital Ocean API token with DNS write access
- Domain's nameservers pointed to Digital Ocean

### 8.1. Set Up Domain on Digital Ocean

1. Go to Digital Ocean → Networking → Domains
2. Add your domain
3. Create an A record:
   - **Hostname**: `vesta` (or `@` for root domain)
   - **Value**: Your droplet's IP address
   - **TTL**: 300 seconds

Wait a few minutes for DNS propagation. Verify:
```bash
dig vesta.yourdomain.com  # Should show your droplet IP
```

### 8.2. Create Digital Ocean API Token

1. Go to Digital Ocean → API → Tokens/Keys
2. Click "Generate New Token"
3. Name: `acme-dns`
4. Scopes: Select "Write"
5. Click "Generate Token"
6. Copy the token (you won't see it again!)

### 8.3. Install acme.sh

```bash
# Install acme.sh
curl https://get.acme.sh | sh -s email=your-email@example.com

# Reload shell
source ~/.bashrc

# Set Digital Ocean API token as environment variable
export DO_API_KEY="your_digital_ocean_api_token_here"
```

### 8.4. Issue SSL Certificate

```bash
# Issue certificate using Digital Ocean DNS validation
acme.sh --issue --dns dns_dgon -d vesta.yourdomain.com --keylength ec-256

# This will:
# - Create a DNS TXT record for validation
# - Wait for Let's Encrypt to verify
# - Download the certificate
# - Store it in ~/.acme.sh/vesta.yourdomain.com_ecc/
```

### 8.5. Install Certificate

```bash
# Create directory for certificates
sudo mkdir -p /etc/vesta/certs
sudo chown -R $USER:$USER /etc/vesta/certs

# Install certificate
acme.sh --install-cert \
  -d vesta.yourdomain.com \
  --ecc \
  --cert-file /etc/vesta/certs/cert.pem \
  --key-file /etc/vesta/certs/key.pem \
  --fullchain-file /etc/vesta/certs/fullchain.pem \
  --reloadcmd "cd ~/vesta && docker-compose restart"
```

### 8.6. Update Docker Configuration

**Update `docker-compose.yml`:**

```yaml
version: '3.8'

services:
  vesta:
    build:
      context: .
      dockerfile: app/Dockerfile
    container_name: vesta
    restart: unless-stopped
    ports:
      - "8000:8000"
      - "443:8000"
    volumes:
      - ./packages/vesta_ml/models:/app/packages/vesta_ml/models:ro
      - ./packages/vesta_ml/data:/app/packages/vesta_ml/data:ro
      - vesta-instance:/app/instance
      - /etc/vesta/certs:/app/certs:ro
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
      - SSL_CERT=/app/certs/fullchain.pem
      - SSL_KEY=/app/certs/key.pem
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('https://localhost:8000/').read()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  vesta-instance:
    driver: local
```

**Update `app/Dockerfile` to use SSL in Gunicorn:**

Add before the CMD line:
```dockerfile
# Run with SSL if certificates are provided
CMD if [ -f "${SSL_CERT}" ] && [ -f "${SSL_KEY}" ]; then \
      gunicorn --bind 0.0.0.0:8000 \
               --certfile="${SSL_CERT}" \
               --keyfile="${SSL_KEY}" \
               --workers 2 --threads 2 --timeout 60 \
               "app:create_app()"; \
    else \
      gunicorn --bind 0.0.0.0:8000 \
               --workers 2 --threads 2 --timeout 60 \
               "app:create_app()"; \
    fi
```

### 8.7. Restart with HTTPS

```bash
cd ~/vesta

# Update firewall
sudo ufw allow 443/tcp
sudo ufw delete allow 8000/tcp  # Remove HTTP if you only want HTTPS

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Test HTTPS
curl https://vesta.yourdomain.com
```

### 8.8. Auto-Renewal

acme.sh automatically sets up a cron job for renewal. Verify:

```bash
crontab -l | grep acme.sh
```

Test renewal:
```bash
acme.sh --renew -d vesta.yourdomain.com --ecc --force
```

### 8.9. Redirect HTTP to HTTPS (Optional)

If you want to support both HTTP and HTTPS with automatic redirect, you'll need nginx. But for simplicity, just use HTTPS only and block port 8000.

---

## 9. Model Management

### Retraining Models

When you improve your models or add more data:

**On the droplet:**
```bash
cd ~/vesta/packages/vesta_ml

# Retrain
python scripts/preprocess.py
python scripts/train.py

# Restart to reload models
cd ~/vesta
docker-compose restart
```

**Or retrain inside the container:**
```bash
docker exec -it vesta bash
cd ../packages/vesta_ml
python scripts/preprocess.py
python scripts/train.py
exit

# Restart
docker-compose restart
```

### Backup Models

Since models aren't in your public GitHub repo, back them up:

```bash
# Create backup directory
mkdir -p ~/backups

# Backup models
cd ~/vesta/packages/vesta_ml
tar czf ~/backups/models-$(date +%Y%m%d).tar.gz models/ data/processed/

# Download to local machine
scp vesta@your_droplet_ip:~/backups/models-*.tar.gz ./local-backups/
```

**Automated backups with cron:**
```bash
crontab -e

# Add this line (runs weekly on Sunday at 2 AM)
0 2 * * 0 cd ~/vesta/packages/vesta_ml && tar czf ~/backups/models-$(date +\%Y\%m\%d).tar.gz models/ data/processed/
```

### How Docker Volumes Work

The `docker-compose.yml` mounts models as read-only:

```yaml
volumes:
  - ./packages/vesta_ml/models:/app/packages/vesta_ml/models:ro
  - ./packages/vesta_ml/data:/app/packages/vesta_ml/data:ro
```

This means:
- Models live on the host at `~/vesta/packages/vesta_ml/models/`
- Docker container reads them from `/app/packages/vesta_ml/models/`
- Changes on the host are immediately available to the container
- Models persist across container restarts and rebuilds

---

## 10. Database Management

### Backup Database

```bash
# Backup SQLite database from Docker volume
docker run --rm \
  -v vesta-instance:/data \
  -v ~/backups:/backup \
  ubuntu \
  tar czf /backup/vesta-db-$(date +%Y%m%d).tar.gz -C /data .

# Or copy directly
docker cp vesta:/app/instance/vesta.db ~/backups/vesta-$(date +%Y%m%d).db
```

### Restore Database

```bash
# Restore from backup
docker cp ~/backups/vesta-20260202.db vesta:/app/instance/vesta.db

# Restart container
docker-compose restart
```

### Automated Database Backups

```bash
crontab -e

# Add this line (daily backup at 2 AM)
0 2 * * * docker cp vesta:/app/instance/vesta.db ~/backups/vesta-$(date +\%Y\%m\%d).db
```

---

## 11. Updating the Application

When you push updates to GitHub:

```bash
cd ~/vesta

# Pull latest code
git pull origin main

# If models changed, retrain
cd packages/vesta_ml
python scripts/preprocess.py
python scripts/train.py
cd ../..

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# View logs to ensure it started correctly
docker-compose logs -f
```

---

## 12. Monitoring and Maintenance

### View Logs

```bash
# Application logs (follow mode)
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs vesta
```

### Resource Monitoring

```bash
# Real-time stats
docker stats vesta

# Disk usage
df -h
docker system df

# Check container status
docker-compose ps
```

### Health Checks

```bash
# Manual health check
curl http://localhost:8000  # or https://vesta.yourdomain.com

# Check if container is healthy
docker inspect vesta --format='{{.State.Health.Status}}'
```

### Automated Health Monitoring

Create `~/check_health.sh`:
```bash
#!/bin/bash
if ! curl -f http://localhost:8000 > /dev/null 2>&1; then
    echo "$(date): Vesta is down! Restarting..." >> ~/health-check.log
    cd ~/vesta && docker-compose restart
fi
```

Make it executable and add to cron:
```bash
chmod +x ~/check_health.sh
crontab -e

# Add: Check every 5 minutes
*/5 * * * * ~/check_health.sh
```

---

## 13. Security Best Practices

### Keep System Updated

```bash
sudo apt update && sudo apt upgrade -y
```

### Enable Automatic Security Updates

```bash
sudo apt install unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

### Secure SSH

Edit `/etc/ssh/sshd_config`:
```bash
sudo nano /etc/ssh/sshd_config
```

Change these settings:
```
PermitRootLogin no
PasswordAuthentication no
```

Restart SSH:
```bash
sudo systemctl restart sshd
```

### Limit Docker Resources

```bash
# Limit memory usage (optional)
docker update --memory="512m" --memory-swap="512m" vesta
```

---

## 14. Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs vesta

# Check Docker daemon
sudo systemctl status docker

# Inspect container
docker inspect vesta
```

### "Model not found" Error

```bash
# Check if models exist on host
ls -la ~/vesta/packages/vesta_ml/models/random_forest/

# If empty, train them
cd ~/vesta/packages/vesta_ml
python scripts/preprocess.py
python scripts/train.py

# Restart
cd ~/vesta
docker-compose restart
```

### Port Already in Use

```bash
# Find what's using the port
sudo lsof -i :8000  # or :443 for HTTPS

# Kill the process or change port in docker-compose.yml
```

### Permission Issues

```bash
# Fix ownership
docker exec -it vesta chown -R root:root /app/instance

# Or on host
sudo chown -R $USER:$USER ~/vesta
```

### SSL Certificate Issues

```bash
# Check certificate validity
acme.sh --list

# Force renewal
acme.sh --renew -d vesta.yourdomain.com --ecc --force

# Check certificate files exist
ls -la /etc/vesta/certs/

# Test SSL connection
openssl s_client -connect vesta.yourdomain.com:443
```

### High Memory Usage

```bash
# Check memory usage
docker stats vesta

# If consistently high, upgrade droplet or optimize:
# - Reduce gunicorn workers in Dockerfile
# - Clear old Docker images: docker system prune -a
```

---

## 15. Cost Estimate

- **Droplet**: $4-6/month
  - $4/month: 512MB RAM, 1 vCPU, 10GB SSD (low traffic)
  - $6/month: 1GB RAM, 1 vCPU, 25GB SSD (recommended)
- **Domain**: ~$10-15/year (optional)
- **SSL Certificate**: Free (Let's Encrypt via acme.sh)

**Total**: $4-7/month

---

## 16. Quick Reference

### Essential Commands

```bash
# Start application
docker-compose up -d

# Stop application
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up -d --build

# Access shell
docker exec -it vesta bash

# Backup database
docker cp vesta:/app/instance/vesta.db ~/backups/

# Update code
git pull && docker-compose up -d --build
```

### File Locations

- **Application code**: `~/vesta/`
- **Models**: `~/vesta/packages/vesta_ml/models/`
- **Database**: Docker volume `vesta-instance` (persistent)
- **SSL certificates**: `/etc/vesta/certs/` (if using HTTPS)
- **Logs**: `docker-compose logs vesta`

---

## Support

For issues:
1. Check application logs: `docker-compose logs vesta`
2. Verify models exist: `ls ~/vesta/packages/vesta_ml/models/random_forest/`
3. Test locally: `curl http://localhost:8000`
4. Check GitHub issues or create a new one

---

**🎉 Congratulations! Your Vesta application is now deployed and running on Digital Ocean.**
