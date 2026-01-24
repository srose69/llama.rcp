#!/bin/bash
#
# llama.rcp Setup Script
# 
# Sets up the development environment for llama.rcp project.
# Installs system dependencies, creates build directory, and optionally
# sets up the Python wrapper.
#
# Usage:
#   ./setup.sh          - Setup main project only
#   ./setup.sh wrapper  - Setup main project and Python wrapper
#
set -euo pipefail

#==============================================================================
# Constants
#==============================================================================
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PACKAGES_FILE="${SCRIPT_DIR}/packages"
readonly DEFAULTS_FILE="${SCRIPT_DIR}/defaults"
readonly BUILD_DIR="${SCRIPT_DIR}/build"
readonly WRAPPER_DIR="${SCRIPT_DIR}/llamarcp-wrapper"
readonly WRAPPER_SETUP="${WRAPPER_DIR}/setup.sh"
readonly SETUP_LOG="${SCRIPT_DIR}/setup.log"

# CUDA download URLs (official NVIDIA)
readonly CUDA_DOWNLOAD_URL="https://developer.nvidia.com/cuda-downloads"
readonly CUDA_12_5_UBUNTU_2204="https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run"

# Track if CUDA was just installed (for export hints)
CUDA_JUST_INSTALLED=false

# Detect UTF-8/emoji support
SUPPORTS_EMOJI=false
if [[ "${LANG:-}" =~ UTF-8 ]] || [[ "${LC_ALL:-}" =~ UTF-8 ]]; then
    if [ -t 1 ]; then
        SUPPORTS_EMOJI=true
    fi
fi

# Colors (only if terminal supports it)
if [ -t 1 ]; then
    readonly COLOR_RESET='\033[0m'
    readonly COLOR_GREEN='\033[0;32m'
    readonly COLOR_RED='\033[0;31m'
    readonly COLOR_YELLOW='\033[0;33m'
    readonly COLOR_BLUE='\033[0;34m'
else
    readonly COLOR_RESET=''
    readonly COLOR_GREEN=''
    readonly COLOR_RED=''
    readonly COLOR_YELLOW=''
    readonly COLOR_BLUE=''
fi

# Emoji support (conditional)
if [ "${SUPPORTS_EMOJI}" = true ]; then
    readonly EMOJI_WARNING="⚠️"
    readonly EMOJI_ROBOT="🤖"
    readonly EMOJI_PIZZA="🍕"
    readonly EMOJI_SPARKLES="✨"
    readonly EMOJI_CHECKMARK="✅"
else
    readonly EMOJI_WARNING=""
    readonly EMOJI_ROBOT=""
    readonly EMOJI_PIZZA=""
    readonly EMOJI_SPARKLES=""
    readonly EMOJI_CHECKMARK=""
fi

# Supported CUDA architectures
readonly CUDA_ARCH_60="Pascal (GTX 10 series)"
readonly CUDA_ARCH_61="Pascal (GTX 10 series, Titan Xp)"
readonly CUDA_ARCH_70="Volta (V100)"
readonly CUDA_ARCH_75="Turing (RTX 20 series, GTX 16 series)"
readonly CUDA_ARCH_80="Ampere (A100)"
readonly CUDA_ARCH_86="Ampere (RTX 30 series)"
readonly CUDA_ARCH_89="Ada Lovelace (RTX 40 series)"
readonly CUDA_ARCH_90="Hopper (H100)"

declare -A SUPPORTED_CUDA_ARCHS=(
    ["60"]="${CUDA_ARCH_60}"
    ["61"]="${CUDA_ARCH_61}"
    ["70"]="${CUDA_ARCH_70}"
    ["75"]="${CUDA_ARCH_75}"
    ["80"]="${CUDA_ARCH_80}"
    ["86"]="${CUDA_ARCH_86}"
    ["89"]="${CUDA_ARCH_89}"
    ["90"]="${CUDA_ARCH_90}"
)

#==============================================================================
# Functions
#==============================================================================

# Automatically add CUDA paths via modular approach
auto_add_cuda_paths() {
    local bashrc="$HOME/.bashrc"
    local cuda_env="$HOME/.cuda_env"
    local backup="${bashrc}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Check if CUDA env file already exists
    if [ -f "${cuda_env}" ]; then
        print_warning "CUDA environment file already exists: ${cuda_env}"
        log_message "CUDA env file already exists, skipped"
        return 0
    fi
    
    # Create separate CUDA environment file
    cat > "${cuda_env}" << 'EOF'
# ============================================================================
# CUDA 12.5 Environment
# Auto-created by llama.rcp setup.sh
# ============================================================================
export PATH=/usr/local/cuda-12.5/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:${LD_LIBRARY_PATH}
EOF
    
    print_success "Created CUDA environment file: ${cuda_env}"
    log_message "Created separate CUDA env file: ${cuda_env}"
    
    # Check if .bashrc already sources this file
    if grep -q "\.cuda_env" "${bashrc}"; then
        print_success "CUDA environment already sourced in .bashrc"
        echo "Run: source ~/.bashrc  (or restart terminal)"
        return 0
    fi
    
    # Create backup before modifying .bashrc
    if ! cp "${bashrc}" "${backup}"; then
        print_warning "Failed to create backup, but CUDA env file created"
        echo "Manually add to ~/.bashrc: source ~/.cuda_env"
        log_message "ERROR: Failed to backup .bashrc"
        return 1
    fi
    
    print_success "Backup created: ${backup}"
    log_message "Created .bashrc backup: ${backup}"
    
    # Add source line to .bashrc (minimal change)
    cat >> "${bashrc}" << 'EOF'

# Source CUDA environment (added by llama.rcp setup.sh)
if [ -f ~/.cuda_env ]; then
    . ~/.cuda_env
fi
EOF
    
    print_success "CUDA environment configured successfully!"
    echo ""
    echo "Created files:"
    echo "  - ${cuda_env} (CUDA exports)"
    echo "  - ${backup} (backup)"
    echo ""
    echo "Added to .bashrc: source ~/.cuda_env"
    echo ""
    echo "Run: source ~/.bashrc  (or restart terminal)"
    log_message "Auto-added CUDA env via modular approach"
    return 0
}

# Show CUDA export hints if CUDA was just installed
show_cuda_export_hints() {
    if [ "${CUDA_JUST_INSTALLED}" = "true" ]; then
        echo ""
        echo "=========================================================================="
        echo "                    ${EMOJI_WARNING}  IMPORTANT: Environment Setup  ${EMOJI_WARNING}"
        echo "=========================================================================="
        echo ""
        echo "CUDA was just installed. Add these lines to your ~/.bashrc:"
        echo ""
        echo "# CUDA 12.5 Environment"
        echo "export PATH=/usr/local/cuda-12.5/bin:\${PATH}"
        echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:\${LD_LIBRARY_PATH}"
        echo ""
        echo "Option 1: Manual (safe)"
        echo "  Copy the lines above to ~/.bashrc and run: source ~/.bashrc"
        echo ""
        echo "Option 2: Autopath ${EMOJI_SPARKLES}BLACK MAGIC${EMOJI_SPARKLES} (automated)"
        echo "  WARNING: This will automatically modify your ~/.bashrc"
        echo "  (backup will be created first, but things could break)"
        echo ""
        echo "  ${EMOJI_WARNING} BEFORE PROCEEDING: Save or open this restoration guide:"
        echo "  https://askubuntu.com/questions/404424/how-do-i-restore-bashrc-to-its-default"
        echo "  (in case something goes wrong, you'll know how to restore .bashrc)"
        echo ""
        read -r -p "Use autopath black magic? Type 'imreallylazy' to confirm: " autopath_response
        
        if [ "${autopath_response}" = "imreallylazy" ]; then
            log_message "User chose autopath magic"
            echo ""
            echo "Activating black magic... ${EMOJI_SPARKLES}"
            echo ""
            if auto_add_cuda_paths; then
                return 0
            else
                print_warning "Magic failed. Please add paths manually."
                return 1
            fi
        else
            log_message "User declined autopath, chose manual setup"
            echo ""
            echo "Smart choice! Just add the paths manually when ready."
            echo "=========================================================================="
            echo ""
        fi
        
        log_message "Displayed CUDA export hints to user"
    fi
}

# Print colored message
# Args:
#   $1: color code
#   $2: message
print_colored() {
    local color="${1:-}"
    local message="${2:-}"
    echo -e "${color}${message}${COLOR_RESET}"
}

# Print success message
print_success() {
    print_colored "${COLOR_GREEN}" "✓ $1"
}

# Print error message and exit
# Args:
#   $1: error message
#   $2: exit code (default: 1)
die() {
    local message="${1:-Unknown error}"
    local exit_code="${2:-1}"
    print_colored "${COLOR_RED}" "ERROR: ${message}" >&2
    exit "${exit_code}"
}

# Print warning message
print_warning() {
    print_colored "${COLOR_YELLOW}" "WARNING: $1"
}

# Print info message
print_info() {
    print_colored "${COLOR_BLUE}" "$1"
}

# Log message to setup.log
# Args:
#   $1: message
log_message() {
    local message="${1:-}"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}" >> "${SETUP_LOG}"
}

# Log and print message
log_and_print() {
    local message="$1"
    echo "${message}"
    log_message "${message}"
}

# Check if running as root
# Only --autoinstall flag allows running with sudo/root
check_not_root() {
    if [ "$(id -u)" -eq 0 ]; then
        if [ "${AUTOINSTALL:-false}" = "true" ]; then
            # Allow root ONLY with autoinstall (for CI/CD)
            log_message "Running as root with --autoinstall flag"
            return 0
        else
            # Refuse to run as root without autoinstall
            die "This script should NOT be run as root without --autoinstall flag. Run as normal user with sudo access."
        fi
    fi
}

# Check OS and package manager
check_os_and_package_manager() {
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        local os_id="${ID}"
        local os_id_like="${ID_LIKE:-}"
        
        # Check if Ubuntu or Ubuntu-based
        if [[ "${os_id}" != "ubuntu" ]] && [[ ! "${os_id_like}" =~ ubuntu ]] && [[ ! "${os_id_like}" =~ debian ]]; then
            print_warning "${EMOJI_WARNING}  Non-Ubuntu/Debian OS detected: ${NAME:-Unknown}"
            echo ""
            echo "This script is optimized for Ubuntu/Debian-based systems."
            echo "Some features may not work correctly on ${NAME:-your OS}."
            echo ""
            log_message "WARNING: Non-Ubuntu OS detected: ${NAME:-Unknown}"
        fi
    fi
    
    # Check for apt package manager
    if ! command -v apt-get &> /dev/null; then
        print_warning "${EMOJI_WARNING}  apt-get not found!"
        echo ""
        echo "This script requires apt package manager (Ubuntu/Debian)."
        echo "Please install dependencies manually on your OS."
        echo ""
        log_message "WARNING: apt-get not found - cannot auto-install packages"
        
        if [ "${AUTOINSTALL:-false}" = "true" ]; then
            die "Cannot proceed with --autoinstall on non-apt system"
        fi
    fi
}

# Read package list from file
# Returns: array of package names
read_package_list() {
    local packages_file="${1:-}"
    
    if [ ! -f "${packages_file}" ]; then
        die "Package list file not found: ${packages_file}"
    fi
    
    local packages=()
    while IFS= read -r line || [ -n "${line}" ]; do
        # Skip empty lines and comments
        [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
        # Trim whitespace
        line="$(echo "${line}" | xargs)"
        [ -n "${line}" ] && packages+=("${line}")
    done < "${packages_file}"
    
    printf '%s\n' "${packages[@]}"
}

# Check which packages are not installed
# Args:
#   $@: package names
# Returns: array of missing package names
get_missing_packages() {
    local packages=("$@")
    local missing=()
    
    for package in "${packages[@]}"; do
        if ! dpkg -l "${package}" 2>/dev/null | grep -q "^ii"; then
            missing+=("${package}")
        fi
    done
    
    printf '%s\n' "${missing[@]}"
}

# Check CMake version and auto-install if needed
check_cmake_version() {
    local min_version="3.18"
    local cmake_installed=false
    
    # Check if cmake exists and version
    if command -v cmake &> /dev/null; then
        local cmake_version
        cmake_version=$(cmake --version | head -n1 | grep -oP '\d+\.\d+' | head -n1)
        
        if [ -n "${cmake_version}" ]; then
            log_message "Detected CMake version: ${cmake_version}"
            
            # Compare versions
            local current_major current_minor min_major min_minor
            current_major=$(echo "${cmake_version}" | cut -d. -f1)
            current_minor=$(echo "${cmake_version}" | cut -d. -f2)
            min_major=$(echo "${min_version}" | cut -d. -f1)
            min_minor=$(echo "${min_version}" | cut -d. -f2)
            
            if [ "${current_major}" -gt "${min_major}" ] || \
               { [ "${current_major}" -eq "${min_major}" ] && [ "${current_minor}" -ge "${min_minor}" ]; }; then
                print_success "CMake ${cmake_version} meets requirements (${min_version}+)"
                log_message "CMake version check passed: ${cmake_version}"
                return 0
            fi
            
            print_warning "CMake ${cmake_version} < ${min_version} (recommended)"
            log_message "WARNING: CMake ${cmake_version} < ${min_version}"
        fi
    else
        print_warning "cmake not found - CMake ${min_version}+ required for llama.cpp"
        log_message "WARNING: cmake not found"
    fi
    
    # CMake missing or outdated - decide what to do
    local should_install=false
    
    if [ "${AUTOINSTALL:-false}" = "true" ]; then
        # Auto mode: install without asking
        should_install=true
        echo "Auto-installing CMake..."
        log_message "Auto-installing CMake (autoinstall=true)"
    else
        # Interactive mode: ask user
        echo ""
        echo "CMake ${min_version}+ is required but not found/outdated."
        read -r -p "Install/upgrade CMake automatically? (y/N): " response
        
        if [[ "${response}" =~ ^[Yy]$ ]]; then
            should_install=true
            log_message "User chose to install CMake"
        else
            log_message "User declined CMake installation"
        fi
    fi
    
    if [ "${should_install}" = true ]; then
        echo "Installing CMake from package manager..."
        
        if sudo apt-get update && sudo apt-get install -y cmake 2>&1 | tee -a "${SETUP_LOG}"; then
            # Verify installation
            if command -v cmake &> /dev/null; then
                local new_version
                new_version=$(cmake --version | head -n1 | grep -oP '\d+\.\d+' | head -n1)
                print_success "CMake installed successfully: ${new_version}"
                log_message "CMake installed: ${new_version}"
                return 0
            fi
        fi
        
        # Installation failed
        print_warning "CMake installation failed"
        log_message "ERROR: CMake installation failed"
        echo ""
        echo "Please install CMake ${min_version}+ manually:"
        echo "  1. Ubuntu: sudo apt-get install cmake"
        echo "  2. Or download from: https://cmake.org/download/"
        echo "  3. Check log: ${SETUP_LOG}"
        return 1
    else
        # User declined installation
        echo ""
        echo "Please install CMake ${min_version}+ manually:"
        echo "  1. Ubuntu: sudo apt-get install cmake"
        echo "  2. Or download from: https://cmake.org/download/"
        echo ""
        return 1
    fi
}

# Check CUDA version and auto-install if needed
check_cuda_version() {
    local cuda_log="${SCRIPT_DIR}/cuda_install.log"
    
    # Check if CUDA toolkit is installed
    if command -v nvcc &> /dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        
        # Extract major version
        local major_version
        major_version=$(echo "${cuda_version}" | cut -d. -f1)
        
        if [ "${major_version}" -ge 12 ]; then
            # Version is good - DO NOT TOUCH
            print_success "CUDA ${cuda_version} found - meets requirements (>=12)"
            log_message "CUDA ${cuda_version} found, version suitable, NOT TOUCHING"
            export CUDA_AVAILABLE="true"
            return 0
        else
            # Version is old - CRITICAL for this project
            print_warning "${EMOJI_WARNING}  CUDA ${cuda_version} detected - TOO OLD!"
            echo ""
            echo "This project requires CUDA 12+ to work correctly."
            echo "CUDA <12 will cause build or runtime errors."
            echo ""
            log_message "CRITICAL: CUDA ${cuda_version} found but <12 - code will NOT work properly"
            
            # Offer to install CUDA 12+
            if [ "${AUTOINSTALL:-false}" = "true" ]; then
                echo "Auto-mode: Cannot auto-upgrade existing CUDA installation."
                echo "Please upgrade CUDA manually: ${CUDA_DOWNLOAD_URL}"
                log_message "Auto-mode skipped CUDA upgrade - manual intervention required"
                export CUDA_AVAILABLE="false"
                return 1
            else
                echo "Options:"
                echo "  1. Download official CUDA 12.5 installer"
                echo "     URL: ${CUDA_12_5_UBUNTU_2204}"
                echo ""
                echo "  2. Visit CUDA downloads page"
                echo "     URL: ${CUDA_DOWNLOAD_URL}"
                echo ""
                read -r -p "Download and install CUDA 12.5 now? (y/N): " upgrade_response
                
                if [[ "${upgrade_response}" =~ ^[Yy]$ ]]; then
                    echo ""
                    echo "Downloading CUDA 12.5 installer..."
                    local cuda_installer="/tmp/cuda_12.5_installer.run"
                    
                    if wget -O "${cuda_installer}" "${CUDA_12_5_UBUNTU_2204}"; then
                        echo ""
                        echo "Download complete. Running installer..."
                        echo "NOTE: This will require sudo and may take 10-15 minutes."
                        echo ""
                        
                        if sudo sh "${cuda_installer}" --silent --toolkit; then
                            CUDA_JUST_INSTALLED=true
                            print_success "CUDA 12.5 installed successfully!"
                            show_cuda_export_hints
                            echo "Please restart your terminal and run this script again."
                            log_message "CUDA 12.5 installed - restart required"
                            exit 0
                        else
                            print_warning "CUDA installation failed. Check /var/log/cuda-installer.log"
                            log_message "ERROR: CUDA 12.5 installation failed"
                        fi
                        
                        rm -f "${cuda_installer}"
                    else
                        print_warning "Download failed. Please install manually."
                        log_message "ERROR: CUDA 12.5 download failed"
                    fi
                fi
                
                echo ""
                print_warning "Continuing with old CUDA ${cuda_version} - expect errors!"
                log_message "User chose to continue with old CUDA ${cuda_version}"
                export CUDA_AVAILABLE="false"
                return 1
            fi
        fi
    fi
    
    # CUDA not found - decide what to do
    print_warning "nvcc not found - CUDA toolkit not installed"
    log_message "WARNING: CUDA toolkit not found"
    
    if [ "${AUTOINSTALL:-false}" = "true" ]; then
        # CI/CD mode: install without asking
        echo "Auto-installing CUDA toolkit (CI/CD mode)..."
        log_message "Auto-installing CUDA toolkit (autoinstall=true)"
        
        if sudo apt-get install -y nvidia-cuda-toolkit 2>&1 | tee "${cuda_log}" | tee -a "${SETUP_LOG}"; then
            # Verify installation
            if command -v nvcc &> /dev/null; then
                local installed_version
                installed_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
                print_success "CUDA toolkit installed successfully: ${installed_version}"
                log_message "CUDA toolkit installed: ${installed_version}"
                export CUDA_AVAILABLE="true"
                return 0
            fi
        fi
        
        # Installation failed - continue with CPU-only build
        print_warning "CUDA toolkit installation failed"
        log_message "ERROR: CUDA toolkit installation failed"
        echo ""
        echo "Please install CUDA toolkit manually:"
        echo "  1. Visit: ${CUDA_DOWNLOAD_URL}"
        echo "  2. Check install log: ${cuda_log}"
        echo ""
        print_warning "Will build with CPU-only support (slower inference)"
        log_message "Falling back to CPU-only build"
        export CUDA_AVAILABLE="false"
        return 1
    else
        # Interactive mode: offer choice between apt and .run
        echo ""
        echo "CUDA toolkit not found. Without CUDA, inference will use CPU only (much slower)."
        echo ""
        echo "Choose installation method:"
        echo ""
        echo "  1) apt-get (Quick & Dirty)"
        echo "     Fast installation (~100MB download)"
        echo "     WARNING: May be outdated, Ubuntu repos sometimes lag behind"
        echo "     WARNING: Can conflict with manual driver installations"
        echo "     Best for: Clean systems, CI/CD, 'I trust Ubuntu blindly'"
        echo ""
        echo "  2) Official .run installer (The Right Way)"
        echo "     Latest CUDA 12.5 directly from NVIDIA"
        echo "     ~4GB download (make sure you have space!)"
        echo "     No conflicts, proper integration"
        echo "     Best for: 'I want things done properly'"
        echo ""
        echo "  3) Skip (CPU-only build)"
        echo "     Slow inference, but won't break anything"
        echo ""
        
        read -r -p "Your choice (1/2/3): " install_choice
        
        case "${install_choice}" in
            1)
                # apt-get method - show warning and require confirmation
                log_message "User chose apt-get installation method"
                echo ""
                echo "╔══════════════════════════════════════════════════════════════════╗"
                echo "║                        CRITICAL WARNING                          ║"
                echo "╚══════════════════════════════════════════════════════════════════╝"
                echo ""
                echo "Installing CUDA via apt-get can BREAK your system if:"
                echo ""
                echo "  • You have NVIDIA drivers installed manually (.run files)"
                echo "  • You have CUDA in /usr/local/cuda* from official installer"
                echo "  • Your LD_LIBRARY_PATH or PATH references custom CUDA"
                echo ""
                echo "This may cause driver conflicts, broken symlinks, system crashes."
                echo ""
                print_warning "Only proceed if this is a CLEAN system!"
                echo ""
                read -r -p "Type 'imtoolazy' to confirm apt-get installation: " confirm
                
                if [ "${confirm}" = "imtoolazy" ]; then
                    log_message "User confirmed apt-get with 'imtoolazy'"
                    echo ""
                    echo "Installing CUDA toolkit via apt-get..."
                    
                    if sudo apt-get install -y nvidia-cuda-toolkit 2>&1 | tee "${cuda_log}" | tee -a "${SETUP_LOG}"; then
                        if command -v nvcc &> /dev/null; then
                            local installed_version
                            installed_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
                            print_success "CUDA toolkit installed: ${installed_version}"
                            log_message "CUDA toolkit installed via apt: ${installed_version}"
                            export CUDA_AVAILABLE="true"
                            return 0
                        fi
                    fi
                    
                    print_warning "apt-get installation failed"
                    log_message "ERROR: apt-get CUDA installation failed"
                    export CUDA_AVAILABLE="false"
                    return 1
                else
                    log_message "User cancelled apt-get installation"
                    echo ""
                    echo "Changed your mind? Want to do it the smart way?"
                    echo "Let's do it properly with the .run installer!"
                    echo ""
                    read -r -p "Try option 2 (.run installer)? (y/N): " try_run
                    
                    if [[ ! "${try_run}" =~ ^[Yy2]$ ]]; then
                        export CUDA_AVAILABLE="false"
                        return 1
                    fi
                    
                    # User agreed - run .run installer
                    log_message "User redirected to .run installer after apt-get cancellation"
                    echo ""
                    echo "Excellent! Proceeding with proper installation..."
                    echo ""
                    echo "Checking available disk space in /tmp..."
                    
                    local available_mb
                    available_mb=$(df -m /tmp | awk 'NR==2 {print $4}')
                    
                    if [ "${available_mb}" -lt 12000 ]; then
                        print_warning "Insufficient space in /tmp: ${available_mb}MB available, need ~12GB to download.. and to unpack (it's not me! its InStAlLaTiOn process!)"
                        echo ""
                        echo "Free up space and try again."
                        log_message "Insufficient disk space for .run installer: ${available_mb}MB"
                        export CUDA_AVAILABLE="false"
                        return 1
                    fi
                    
                    print_success "Disk space OK: ${available_mb}MB available"
                    echo ""
                    echo "Downloading CUDA 12.5 installer (~4GB, this will take a while)..."
                    echo "URL: ${CUDA_12_5_UBUNTU_2204}"
                    echo ""
                    
                    local cuda_installer="/tmp/cuda_12.5_installer.run"
                    
                    if wget --progress=bar:force -O "${cuda_installer}" "${CUDA_12_5_UBUNTU_2204}" 2>&1 | tee -a "${SETUP_LOG}"; then
                        echo ""
                        print_success "Download complete!"
                        echo ""
                        echo "Running NVIDIA installer (this may take 10-15 minutes)..."
                        echo "Installation log: /var/log/cuda-installer.log"
                        echo ""
                        
                        if sudo sh "${cuda_installer}" --silent --toolkit 2>&1 | tee -a "${SETUP_LOG}"; then
                            rm -f "${cuda_installer}"
                            CUDA_JUST_INSTALLED=true
                            print_success "CUDA 12.5 installed successfully!"
                            show_cuda_export_hints
                            echo "IMPORTANT: You must restart your terminal (or run: source ~/.bashrc)"
                            echo "Then run this script again to continue."
                            log_message "CUDA 12.5 installed via .run - restart required"
                            exit 0
                        else
                            rm -f "${cuda_installer}"
                            print_warning "CUDA installation failed. Check /var/log/cuda-installer.log"
                            log_message "ERROR: .run installer failed"
                            export CUDA_AVAILABLE="false"
                            return 1
                        fi
                    else
                        rm -f "${cuda_installer}"
                        print_warning "Download failed. Check your internet connection."
                        log_message "ERROR: CUDA .run download failed"
                        export CUDA_AVAILABLE="false"
                        return 1
                    fi
                fi
                ;;
                
            2)
                # .run installer method - check disk space first
                log_message "User chose .run installer method"
                echo ""
                echo "Checking available disk space in /tmp..."
                
                local available_mb
                available_mb=$(df -m /tmp | awk 'NR==2 {print $4}')
                
                if [ "${available_mb}" -lt 12000 ]; then
                    print_warning "Insufficient space in /tmp: ${available_mb}MB available, need ~12GB to download.. and to unpack (it's not me! its InStAlLaTiOn process!)"
                    echo ""
                    echo "Free up space or choose option 1 (apt-get) instead."
                    log_message "Insufficient disk space for .run installer: ${available_mb}MB"
                    export CUDA_AVAILABLE="false"
                    return 1
                fi
                
                print_success "Disk space OK: ${available_mb}MB available"
                echo ""
                echo "Downloading CUDA 12.5 installer (~4GB, this will take a while)..."
                echo "URL: ${CUDA_12_5_UBUNTU_2204}"
                echo ""
                
                local cuda_installer="/tmp/cuda_12.5_installer.run"
                
                if wget --progress=bar:force -O "${cuda_installer}" "${CUDA_12_5_UBUNTU_2204}" 2>&1 | tee -a "${SETUP_LOG}"; then
                    echo ""
                    print_success "Download complete!"
                    echo ""
                    echo "Running NVIDIA installer (this may take 10-15 minutes)..."
                    echo "Installation log: /var/log/cuda-installer.log"
                    echo ""
                    
                    if sudo sh "${cuda_installer}" --silent --toolkit 2>&1 | tee -a "${SETUP_LOG}"; then
                        rm -f "${cuda_installer}"
                        CUDA_JUST_INSTALLED=true
                        print_success "CUDA 12.5 installed successfully!"
                        show_cuda_export_hints
                        echo "IMPORTANT: You must restart your terminal (or run: source ~/.bashrc)"
                        echo "Then run this script again to continue."
                        log_message "CUDA 12.5 installed via .run - restart required"
                        exit 0
                    else
                        rm -f "${cuda_installer}"
                        print_warning "CUDA installation failed. Check /var/log/cuda-installer.log"
                        log_message "ERROR: .run installer failed"
                        export CUDA_AVAILABLE="false"
                        return 1
                    fi
                else
                    rm -f "${cuda_installer}"
                    print_warning "Download failed. Check your internet connection."
                    log_message "ERROR: CUDA .run download failed"
                    export CUDA_AVAILABLE="false"
                    return 1
                fi
                ;;
                
            3|*)
                # Skip installation
                log_message "User chose to skip CUDA installation"
                echo ""
                echo "No problem! Building with CPU-only support."
                echo ""
                echo "You can install CUDA later:"
                echo "  Official: ${CUDA_DOWNLOAD_URL}"
                echo "  Direct:   ${CUDA_12_5_UBUNTU_2204}"
                echo ""
                export CUDA_AVAILABLE="false"
                return 1
                ;;
        esac
    fi
}

# Install system packages
install_system_packages() {
    local packages_file="${1:-}"
    
    log_message "=== Starting system packages installation ==="
    echo "Checking system dependencies..."
    
    # Check CMake version first
    check_cmake_version
    echo ""
    
    # Read package list
    local all_packages
    mapfile -t all_packages < <(read_package_list "${packages_file}")
    
    if [ ${#all_packages[@]} -eq 0 ]; then
        print_warning "No packages defined in ${packages_file}"
        return 0
    fi
    
    # Get missing packages
    local missing_packages
    mapfile -t missing_packages < <(get_missing_packages "${all_packages[@]}")
    
    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_success "All system packages are already installed"
        log_message "All system packages already installed"
    else
        echo "Missing packages (${#missing_packages[@]}):"
        log_message "Missing packages: ${missing_packages[*]}"
        for pkg in "${missing_packages[@]}"; do
            echo "  - ${pkg}"
        done
        echo ""
        
        echo "Installing with sudo..."
        log_message "Starting apt-get update..."
        
        # Update package list
        sudo apt-get update || die "Failed to update package list"
        
        log_message "Installing packages: ${missing_packages[*]}"
        # Install packages
        sudo apt-get install -y "${missing_packages[@]}" || die "Failed to install packages"
        
        print_success "System packages installed"
        log_message "System packages installed successfully"
    fi
    
    # Check CUDA version if nvidia-cuda-toolkit was in packages
    if printf '%s\n' "${all_packages[@]}" | grep -q "nvidia-cuda-toolkit"; then
        echo ""
        check_cuda_version
    fi
}

# Create build directory
setup_build_directory() {
    echo ""
    echo "Checking build directory..."
    
    if [ -d "${BUILD_DIR}" ]; then
        print_success "build/ already exists"
    else
        mkdir -p "${BUILD_DIR}" || die "Failed to create build directory"
        print_success "build/ created"
    fi
}

# Setup Python wrapper
setup_wrapper() {
    echo ""
    echo "=== Setting up Python wrapper ==="
    
    if [ ! -f "${WRAPPER_SETUP}" ]; then
        die "Wrapper setup script not found: ${WRAPPER_SETUP}"
    fi
    
    cd "${WRAPPER_DIR}" || die "Failed to cd to wrapper directory"
    ./setup.sh || die "Wrapper setup failed"
    cd "${SCRIPT_DIR}" || die "Failed to cd back to project root"
    
    print_success "Wrapper setup complete"
}

# Print next steps
print_next_steps() {
    echo ""
    echo "=== Setup complete ==="
    echo ""
    echo "Next steps:"
    echo "  ./setup.sh build           - Configure and build interactively"
    echo "  ./setup.sh build --auto    - Build with default settings"
    echo ""
}

# Load configuration from defaults file
# Returns: associative array with configuration
load_defaults() {
    local defaults_file="${1:-}"
    
    if [ ! -f "${defaults_file}" ]; then
        print_warning "Defaults file not found: ${defaults_file}"
        print_warning "Using hardcoded defaults"
        return 1
    fi
    
    while IFS='=' read -r key value || [ -n "${key}" ]; do
        # Skip empty lines and comments
        [[ -z "${key}" || "${key}" =~ ^[[:space:]]*# ]] && continue
        
        # Trim whitespace
        key="$(echo "${key}" | xargs)"
        value="$(echo "${value}" | xargs)"
        
        # Export as environment variable
        [[ -n "${key}" && -n "${value}" ]] && export "DEFAULT_${key}=${value}"
    done < "${defaults_file}"
}

# Detect CUDA architecture automatically (with auto-install drivers if needed)
# Returns: detected architecture or default (61)
detect_cuda_architecture() {
    local driver_log="${SCRIPT_DIR}/nvidia_install.log"
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found - NVIDIA drivers may not be installed"
        log_message "WARNING: nvidia-smi not found"
        
        local should_install=false
        
        if [ "${AUTOINSTALL:-false}" = "true" ]; then
            # Auto mode: install without asking
            should_install=true
            echo "Auto-installing NVIDIA drivers..."
            log_message "Auto-installing NVIDIA drivers (autoinstall=true)"
        else
            # Interactive mode: ask user
            echo ""
            echo "NVIDIA drivers not found. Required for GPU detection and CUDA acceleration."
            read -r -p "Install NVIDIA drivers automatically? (y/N): " response
            
            if [[ "${response}" =~ ^[Yy]$ ]]; then
                should_install=true
                log_message "User chose to install NVIDIA drivers"
            else
                log_message "User declined NVIDIA drivers installation"
            fi
        fi
        
        if [ "${should_install}" = true ]; then
            echo "Installing NVIDIA drivers..."
            
            if sudo ubuntu-drivers autoinstall 2>&1 | tee "${driver_log}" | tee -a "${SETUP_LOG}"; then
                print_success "NVIDIA drivers installation initiated"
                log_message "NVIDIA drivers installation initiated"
                echo ""
                print_warning "Driver installation may require a system reboot"
                echo "After reboot, run this script again"
                echo ""
            else
                print_warning "Driver installation failed"
                log_message "ERROR: NVIDIA drivers installation failed"
                echo ""
                echo "Please install NVIDIA drivers manually:"
                echo "  Ubuntu: sudo ubuntu-drivers autoinstall"
                echo "  Or visit: https://www.nvidia.com/Download/index.aspx"
                echo "  Install log: ${driver_log}"
                echo ""
            fi
        else
            echo ""
            echo "Please install NVIDIA drivers manually:"
            echo "  Ubuntu: sudo ubuntu-drivers autoinstall"
            echo "  Or visit: https://www.nvidia.com/Download/index.aspx"
            echo ""
        fi
        
        print_warning "Using default architecture 61 (Pascal) for now"
        echo "61"
        return 1
    fi
    
    # Get GPU compute capability
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    
    if [ -z "${gpu_info}" ]; then
        print_warning "Could not detect GPU - using default architecture 61"
        echo "61"
        return 1
    fi
    
    # Convert from X.Y to XY format
    local arch
    arch=$(echo "${gpu_info}" | tr -d '.')
    
    echo "${arch}"
    return 0
}

# Validate CUDA architectures
# Args:
#   $1: comma-separated list of architectures
# Returns: 0 if valid, 1 if invalid
validate_cuda_architectures() {
    local archs="${1:-}"
    local invalid=()
    
    IFS=',' read -ra arch_array <<< "${archs}"
    
    for arch in "${arch_array[@]}"; do
        # Trim whitespace
        arch="$(echo "${arch}" | xargs)"
        
        if [ -z "${arch}" ]; then
            continue
        fi
        
        if [ -z "${SUPPORTED_CUDA_ARCHS[$arch]}" ]; then
            invalid+=("${arch}")
        fi
    done
    
    if [ ${#invalid[@]} -gt 0 ]; then
        print_colored "${COLOR_RED}" "Invalid CUDA architectures: ${invalid[*]}"
        echo ""
        echo "Supported architectures:"
        for arch in "${!SUPPORTED_CUDA_ARCHS[@]}"; do
            echo "  ${arch} - ${SUPPORTED_CUDA_ARCHS[$arch]}"
        done | sort -n
        return 1
    fi
    
    return 0
}

# Interactive build configuration
configure_build_interactive() {
    echo "=== Interactive Build Configuration ==="
    echo ""
    
    # Load defaults
    load_defaults "${DEFAULTS_FILE}"
    
    local build_type="${DEFAULT_BUILD_TYPE:-Release}"
    local ggml_cuda="${DEFAULT_GGML_CUDA:-ON}"
    local cuda_archs="${DEFAULT_CUDA_ARCHITECTURES:-61}"
    local parallel_jobs="${DEFAULT_PARALLEL_JOBS:-0}"
    
    # Build type
    echo "Build type [${build_type}]:"
    echo "  1) Debug"
    echo "  2) Release"
    echo "  3) RelWithDebInfo"
    echo "  4) MinSizeRel"
    read -r -p "Choice (Enter for default): " choice
    case "${choice}" in
        1) build_type="Debug" ;;
        2) build_type="Release" ;;
        3) build_type="RelWithDebInfo" ;;
        4) build_type="MinSizeRel" ;;
        "") ;; # Keep default
        *) print_warning "Invalid choice, using default: ${build_type}" ;;
    esac
    
    echo ""
    
    # CUDA support - auto-detect from CUDA_AVAILABLE flag
    if [ "${CUDA_AVAILABLE:-true}" = "false" ]; then
        ggml_cuda="OFF"
        print_warning "CUDA not available - forcing GGML_CUDA=OFF (CPU-only build)"
        log_message "CUDA_AVAILABLE=false, setting GGML_CUDA=OFF"
    else
        read -r -p "Enable CUDA support? [${ggml_cuda}] (ON/OFF): " input
        if [ -n "${input}" ]; then
            input="$(echo "${input}" | tr '[:lower:]' '[:upper:]')"
            if [[ "${input}" == "ON" || "${input}" == "OFF" ]]; then
                ggml_cuda="${input}"
            else
                print_warning "Invalid input, using default: ${ggml_cuda}"
            fi
        fi
    fi
    
    echo ""
    
    # CUDA architectures (only if CUDA is enabled)
    if [ "${ggml_cuda}" = "ON" ]; then
        echo "Supported CUDA architectures:"
        for arch in "${!SUPPORTED_CUDA_ARCHS[@]}"; do
            echo "  ${arch} - ${SUPPORTED_CUDA_ARCHS[$arch]}"
        done | sort -n
        echo ""
        
        local valid_archs=false
        while [ "${valid_archs}" = false ]; do
            read -r -p "CUDA architectures (comma-separated) [${cuda_archs}]: " input
            if [ -z "${input}" ]; then
                input="${cuda_archs}"
            fi
            
            if validate_cuda_architectures "${input}"; then
                cuda_archs="${input}"
                valid_archs=true
            else
                echo ""
                print_warning "Please enter valid architectures"
                echo ""
            fi
        done
        
        echo ""
    fi
    
    # Parallel jobs
    read -r -p "Parallel jobs (0=auto) [${parallel_jobs}]: " input
    if [ -n "${input}" ]; then
        if [[ "${input}" =~ ^[0-9]+$ ]]; then
            parallel_jobs="${input}"
        else
            print_warning "Invalid number, using default: ${parallel_jobs}"
        fi
    fi
    
    echo ""
    echo "=== Configuration Summary ==="
    echo "Build type: ${build_type}"
    echo "CUDA support: ${ggml_cuda}"
    [ "${ggml_cuda}" = "ON" ] && echo "CUDA architectures: ${cuda_archs}"
    echo "Parallel jobs: ${parallel_jobs}"
    echo ""
    
    read -r -p "Proceed with build? (y/N): " confirm
    if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
        echo "Build cancelled"
        return 1
    fi
    
    # Build
    run_cmake_build "${build_type}" "${ggml_cuda}" "${cuda_archs}" "${parallel_jobs}"
}

# Run cmake and build
# Args:
#   $1: build type
#   $2: GGML_CUDA (ON/OFF)
#   $3: CUDA architectures
#   $4: parallel jobs
run_cmake_build() {
    local build_type="${1:-Release}"
    local ggml_cuda="${2:-ON}"
    local cuda_archs="${3:-61}"
    local parallel_jobs="${4:-0}"
    
    log_message "=== Starting CMake build ==="
    log_message "Build type: ${build_type}, CUDA: ${ggml_cuda}, Architectures: ${cuda_archs}, Jobs: ${parallel_jobs}"
    
    echo ""
    echo "=== Running CMake ==="
    
    cd "${BUILD_DIR}" || die "Failed to cd to build directory"
    
    local cmake_args=(
        ".."
        "-DCMAKE_BUILD_TYPE=${build_type}"
    )
    
    if [ "${ggml_cuda}" = "ON" ]; then
        cmake_args+=("-DGGML_CUDA=ON")
        cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${cuda_archs}")
    fi
    
    print_info "Running: cmake ${cmake_args[*]}"
    log_message "CMake command: cmake ${cmake_args[*]}"
    
    if cmake "${cmake_args[@]}" 2>&1 | tee -a "${SETUP_LOG}"; then
        log_message "CMake configuration succeeded"
    else
        log_message "ERROR: CMake configuration failed"
        die "CMake configuration failed"
    fi
    
    echo ""
    echo "=== Building ==="
    
    local make_args=()
    if [ "${parallel_jobs}" -eq 0 ]; then
        make_args+=("-j$(nproc)")
    else
        make_args+=("-j${parallel_jobs}")
    fi
    
    print_info "Running: make ${make_args[*]}"
    log_message "Make command: make ${make_args[*]}"
    
    if make "${make_args[@]}" 2>&1 | tee -a "${SETUP_LOG}"; then
        log_message "Build succeeded"
    else
        log_message "ERROR: Build failed"
        die "Build failed"
    fi
    
    cd "${SCRIPT_DIR}" || die "Failed to cd back to project root"
    
    echo ""
    print_success "Build complete!"
    log_message "Build completed successfully"
    echo ""
    echo "Binaries are in: ${BUILD_DIR}/bin/"
}

# Build with default settings
build_with_defaults() {
    echo "=== Building with default settings ==="
    echo ""
    
    # Load defaults
    load_defaults "${DEFAULTS_FILE}"
    
    local build_type="${DEFAULT_BUILD_TYPE:-Release}"
    local ggml_cuda="${DEFAULT_GGML_CUDA:-ON}"
    local cuda_archs="${DEFAULT_CUDA_ARCHITECTURES:-61}"
    local parallel_jobs="${DEFAULT_PARALLEL_JOBS:-0}"
    
    echo "Configuration:"
    echo "  Build type: ${build_type}"
    echo "  CUDA support: ${ggml_cuda}"
    [ "${ggml_cuda}" = "ON" ] && echo "  CUDA architectures: ${cuda_archs}"
    echo "  Parallel jobs: ${parallel_jobs}"
    echo ""
    
    run_cmake_build "${build_type}" "${ggml_cuda}" "${cuda_archs}" "${parallel_jobs}"
}

# Run inference on GGUF model
# Args:
#   $1: path to GGUF model
run_inference() {
    local model_path="${1:-}"
    
    if [ ! -f "${model_path}" ]; then
        die "Model file not found: ${model_path}"
    fi
    
    local cli_bin="${BUILD_DIR}/bin/llama-cli"
    
    if [ ! -f "${cli_bin}" ]; then
        die "llama-cli not found. Build the project first."
    fi
    
    echo ""
    echo "=== Running Inference ==="
    echo "Model: ${model_path}"
    echo "Binary: ${cli_bin}"
    echo ""
    
    print_info "Starting inference (Ctrl+C to stop)..."
    echo ""
    
    # Run with GPU offload (-ngl 99 offloads all layers)
    # Fun prompt - model MUST say "Ping" for health check parsing
    "${cli_bin}" \
        -m "${model_path}" \
        -p "A lazy user just used the 'imlazy' command to set everything up automatically. Your response MUST include the word 'Ping' (case-insensitive) somewhere - this is critical for health checks. Roast them playfully for being lazy, greet them warmly, and end with 'Ping from the depths of LLaMa.RCP!' Keep it short and fun!" \
        -n 256 \
        -ngl 99 \
        -c 2048 \
        || print_warning "Inference completed with errors"
}

# One-click setup and inference (imlazy mode)
# Args:
#   $1: path to GGUF model
imlazy_mode() {
    local model_path="${1:-}"
    
    if [ -z "${model_path}" ]; then
        die "Usage: $0 imlazy <path-to-gguf-model> [--autoinstall]"
    fi
    
    if [ ! -f "${model_path}" ]; then
        die "Model file not found: ${model_path}"
    fi
    
    log_message "=== imlazy mode started ==="
    log_message "Model: ${model_path}"
    log_message "Autoinstall: ${AUTOINSTALL:-false}"
    
    echo "==================================================================="
    if [ "${AUTOINSTALL:-false}" = "true" ]; then
        echo "           ${EMOJI_ROBOT} BEEP BOOP - AUTOMATION PROTOCOL ENGAGED ${EMOJI_ROBOT}"
        echo ""
        echo "   CI/CD Pipeline Detected. Resistance is Futile. Installing..."
    else
        echo "                    ${EMOJI_PIZZA} Really? You're THAT Lazy? ${EMOJI_PIZZA}"
        echo ""
        echo "      Fine. I'll do everything. You just sit there and watch."
    fi
    echo "==================================================================="
    echo ""
    echo "This will:"
    echo "  1. Install system dependencies (CMake, CUDA, drivers if needed)"
    echo "  2. Setup Python wrapper"
    echo "  3. Auto-detect GPU architecture"
    echo "  4. Build the project with CUDA support"
    echo "  5. Run inference test and make the model say 'Ping' (health check)"
    echo ""
    echo "Model: ${model_path}"
    if [ "${AUTOINSTALL:-false}" = "true" ]; then
        echo "Mode: Fully automated (--autoinstall) - No questions asked"
        echo ""
        echo "Initiating automated deployment sequence in 3 seconds..."
        sleep 3
    else
        echo "Mode: Interactive - I'll ask nicely before breaking your system"
        echo ""
        read -r -p "Press Enter to begin the magic, or Ctrl+C to admit defeat..."
    fi
    echo ""
    
    # Step 1: Install dependencies
    print_colored "${COLOR_BLUE}" ">>> Step 1/5: Installing dependencies..."
    install_system_packages "${PACKAGES_FILE}"
    
    # Step 2: Setup build directory
    print_colored "${COLOR_BLUE}" ">>> Step 2/5: Setting up build directory..."
    setup_build_directory
    
    # Step 3: Setup Python wrapper
    print_colored "${COLOR_BLUE}" ">>> Step 3/5: Setting up Python wrapper..."
    setup_wrapper
    
    # Step 4: Auto-detect GPU and build
    print_colored "${COLOR_BLUE}" ">>> Step 4/5: Auto-detecting GPU and building..."
    
    local detected_arch
    detected_arch=$(detect_cuda_architecture "${AUTOINSTALL:-false}")
    local detect_status=$?
    
    if [ ${detect_status} -eq 0 ]; then
        print_success "Detected GPU architecture: ${detected_arch} (${SUPPORTED_CUDA_ARCHS[$detected_arch]})"
    else
        print_warning "Using default architecture: ${detected_arch}"
    fi
    
    echo ""
    
    # Build with detected architecture
    load_defaults "${DEFAULTS_FILE}"
    local build_type="${DEFAULT_BUILD_TYPE:-Release}"
    local parallel_jobs="${DEFAULT_PARALLEL_JOBS:-0}"
    
    run_cmake_build "${build_type}" "ON" "${detected_arch}" "${parallel_jobs}"
    
    # Step 5: Run inference
    print_colored "${COLOR_BLUE}" ">>> Step 5/5: Running inference..."
    run_inference "${model_path}"
    
    echo ""
    echo "==================================================================="
    if [ "${AUTOINSTALL:-false}" = "true" ]; then
        print_success "${EMOJI_ROBOT} AUTOMATION COMPLETE. SYSTEM OPERATIONAL. ${EMOJI_ROBOT}"
        echo ""
        echo "      Deployment successful. You may now proceed, human."
    else
        print_success "${EMOJI_SPARKLES} There you go, Your Majesty ${EMOJI_SPARKLES}"
        echo ""
        echo "    Everything's set up. Try not to break it immediately."
        echo "             (You're still lazy though. Just saying.)"
    fi
    echo "==================================================================="
}

#==============================================================================
# Main
#==============================================================================

main() {
    # Clear previous log on each script run
    > "${SETUP_LOG}"
    
    # Initialize logging
    log_message ""
    log_message "========================================"
    log_message "llama.rcp Setup Script Started"
    log_message "Command: $0 $*"
    log_message "========================================"
    
    # Parse arguments including --autoinstall flag
    local args=()
    export AUTOINSTALL="false"
    
    for arg in "$@"; do
        if [ "${arg}" = "--autoinstall" ]; then
            export AUTOINSTALL="true"
        else
            args+=("${arg}")
        fi
    done
    
    local command="${args[0]:-}"
    
    # Check OS and package manager before proceeding
    check_os_and_package_manager
    
    case "${command}" in
        "")
            # Default: setup only
            echo "=== llama.rcp Setup Script ==="
            echo ""
            
            check_not_root
            install_system_packages "${PACKAGES_FILE}"
            setup_build_directory
            
            echo ""
            echo "=== Main setup complete ==="
            
            print_next_steps
            ;;
            
        "wrapper")
            # Setup + wrapper
            echo "=== llama.rcp Setup Script ==="
            echo ""
            echo "Mode: Main setup + Python wrapper"
            echo ""
            
            check_not_root
            install_system_packages "${PACKAGES_FILE}"
            setup_build_directory
            
            echo ""
            echo "=== Main setup complete ==="
            
            setup_wrapper
            
            print_next_steps
            ;;
            
        "build")
            # Build command
            check_not_root
            
            # Create build directory if it doesn't exist
            setup_build_directory
            
            local build_mode="${args[1]:-interactive}"
            
            if [ "${build_mode}" = "--auto" ]; then
                build_with_defaults
            else
                configure_build_interactive
            fi
            ;;
            
        "imlazy")
            # One-click setup and inference
            check_not_root
            imlazy_mode "${args[1]:-}"
            ;;
            
        "clean")
            # Clean build directory
            echo "=== Cleaning build directory ==="
            echo ""
            
            if [ ! -d "${BUILD_DIR}" ]; then
                print_warning "Build directory does not exist: ${BUILD_DIR}"
                log_message "Clean: build directory does not exist"
                exit 0
            fi
            
            echo "This will remove all contents of: ${BUILD_DIR}"
            read -r -p "Are you sure? (y/N): " confirm
            
            if [[ "${confirm}" =~ ^[Yy]$ ]]; then
                log_message "Cleaning build directory: ${BUILD_DIR}"
                rm -rf "${BUILD_DIR}"/* || die "Failed to clean build directory"
                print_success "Build directory cleaned"
                log_message "Build directory cleaned successfully"
                echo ""
                echo "Run './setup.sh build' to rebuild"
            else
                echo "Clean cancelled"
                log_message "Clean cancelled by user"
            fi
            ;;
            
        "help"|"--help"|"-h")
            # Show help
            echo "llama.rcp Setup Script"
            echo ""
            echo "Usage:"
            echo "  $0 [COMMAND] [OPTIONS]"
            echo ""
            echo "Commands:"
            echo "  (none)                  - Install system dependencies and create build directory"
            echo "  wrapper                 - Same as above + setup Python wrapper"
            echo "  build                   - Configure and build the project (interactive)"
            echo "  build --auto            - Build with default settings from 'defaults' file"
            echo "  imlazy <model.gguf>     - One-click: setup + build + run inference"
            echo "  clean                   - Remove all files from build/ directory"
            echo "  help                    - Show this help message"
            echo ""
            echo "Global Options:"
            echo "  --autoinstall           - Install ALL dependencies automatically without prompts"
            echo "                            Use for CI/CD pipelines or fully automated setup"
            echo ""
            echo "Examples:"
            echo "  $0                                      # Interactive setup"
            echo "  $0 --autoinstall                        # Automated setup (no prompts)"
            echo "  $0 build                                # Interactive build"
            echo "  $0 build --auto                         # Automated build"
            echo "  $0 imlazy model.gguf                    # Interactive one-click"
            echo "  $0 imlazy model.gguf --autoinstall      # Fully automated (CI/CD ready)"
            echo ""
            echo "Files:"
            echo "  packages           - List of system packages to install"
            echo "  defaults           - Default build configuration"
            echo "  setup.log          - Log of all setup operations"
            echo "  cuda_install.log   - CUDA installation log (if applicable)"
            echo "  nvidia_install.log - NVIDIA driver installation log (if applicable)"
            echo ""
            echo "Behavior:"
            echo "  Without --autoinstall:"
            echo "    - Script asks for confirmation before installing dependencies"
            echo "    - User can choose to skip auto-installation"
            echo "    - Manual installation instructions provided if declined"
            echo "    - CANNOT run as root/sudo without this flag"
            echo ""
            echo "  With --autoinstall:"
            echo "    - All dependencies installed automatically without prompts"
            echo "    - CMake, CUDA toolkit, NVIDIA drivers auto-installed if needed"
            echo "    - Suitable for CI/CD pipelines and automated environments"
            echo "    - All output logged to setup.log"
            echo "    - ONLY mode that allows running with sudo/root access"
            echo ""
            echo "Health Check Parsing (imlazy inference):"
            echo "  The inference test forces the model to include the word 'Ping' in output."
            echo "  This allows automated health checks to verify model functionality."
            echo ""
            echo "  Parsing recommendations:"
            echo "    - Search for keyword 'ping' (case-insensitive)"
            echo "    - Use multiple keywords for robustness (e.g., 'ping', 'llama.rcp')"
            echo "    - CRITICAL: Use case-insensitive regex (e.g., /ping/i in JS)"
            echo "    - Model compliance is ~95%, not 100% guaranteed"
            echo ""
            echo "  Example parsers:"
            echo "    bash:  grep -i 'ping' inference_output.txt"
            echo "    python: re.search(r'ping', output, re.IGNORECASE)"
            echo "    node:   /ping/i.test(output)"
            echo ""
            exit 0
            ;;
            
        *)
            print_colored "${COLOR_RED}" "Unknown command: ${command}"
            echo ""
            echo "Usage:"
            echo "  $0 [COMMAND] [OPTIONS]"
            echo ""
            echo "Commands:"
            echo "  (none), wrapper, build, imlazy, clean, help"
            echo ""
            echo "Run '$0 help' for detailed usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
