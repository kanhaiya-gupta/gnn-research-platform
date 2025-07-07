// GNN Platform Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Add loading states to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.classList.contains('btn-loading')) {
                return; // Prevent multiple clicks
            }
            
            const originalText = this.innerHTML;
            this.innerHTML = '<span class="loading"></span> Loading...';
            this.classList.add('btn-loading');
            this.disabled = true;
            
            // Reset after 3 seconds (for demo purposes)
            setTimeout(() => {
                this.innerHTML = originalText;
                this.classList.remove('btn-loading');
                this.disabled = false;
            }, 3000);
        });
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add hover effects to cards
    const cards = document.querySelectorAll('.card, .purpose-card, .stat-card, .research-card, .model-card, .step-card, .feature-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 25px rgba(0,0,0,0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 5px 15px rgba(0,0,0,0.1)';
        });
    });

    // Copy to clipboard functionality
    document.querySelectorAll('.copy-btn').forEach(button => {
        button.addEventListener('click', function() {
            const text = this.getAttribute('data-clipboard-text');
            if (text) {
                navigator.clipboard.writeText(text).then(() => {
                    // Show success message
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    this.classList.add('btn-success');
                    
                    setTimeout(() => {
                        this.innerHTML = originalText;
                        this.classList.remove('btn-success');
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                });
            }
        });
    });

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });

    // Dynamic parameter updates
    const parameterInputs = document.querySelectorAll('.parameter-input');
    parameterInputs.forEach(input => {
        input.addEventListener('change', function() {
            updateParameterDisplay(this);
        });
    });

    // Auto-save functionality
    let autoSaveTimeout;
    const autoSaveInputs = document.querySelectorAll('.auto-save');
    autoSaveInputs.forEach(input => {
        input.addEventListener('input', function() {
            clearTimeout(autoSaveTimeout);
            autoSaveTimeout = setTimeout(() => {
                saveToLocalStorage(this);
            }, 1000);
        });
    });

    // Load saved data on page load
    loadFromLocalStorage();

    // Initialize charts if Chart.js is available
    if (typeof Chart !== 'undefined') {
        initializeCharts();
    }

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit forms
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const activeForm = document.querySelector('form:focus-within');
            if (activeForm) {
                const submitBtn = activeForm.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.click();
                }
            }
        }
        
        // Escape key to close modals
        if (e.key === 'Escape') {
            const openModal = document.querySelector('.modal.show');
            if (openModal) {
                const closeBtn = openModal.querySelector('.btn-close');
                if (closeBtn) {
                    closeBtn.click();
                }
            }
        }
    });

    // Add search functionality
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const searchableElements = document.querySelectorAll('.searchable');
            
            searchableElements.forEach(element => {
                const text = element.textContent.toLowerCase();
                if (text.includes(searchTerm)) {
                    element.style.display = '';
                } else {
                    element.style.display = 'none';
                }
            });
        });
    }

    // Add dark mode toggle
    const darkModeToggle = document.querySelector('.dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            const isDarkMode = document.body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDarkMode);
            
            // Update toggle icon
            const icon = this.querySelector('i');
            if (isDarkMode) {
                icon.className = 'fas fa-sun';
            } else {
                icon.className = 'fas fa-moon';
            }
        });
    }

    // Load dark mode preference
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode === 'true') {
        document.body.classList.add('dark-mode');
        const icon = document.querySelector('.dark-mode-toggle i');
        if (icon) {
            icon.className = 'fas fa-sun';
        }
    }
});

// Helper functions
function updateParameterDisplay(input) {
    const displayElement = document.querySelector(`#${input.id}-display`);
    if (displayElement) {
        displayElement.textContent = input.value;
    }
}

function saveToLocalStorage(input) {
    const key = `gnn_platform_${input.name}`;
    localStorage.setItem(key, input.value);
}

function loadFromLocalStorage() {
    const autoSaveInputs = document.querySelectorAll('.auto-save');
    autoSaveInputs.forEach(input => {
        const key = `gnn_platform_${input.name}`;
        const savedValue = localStorage.getItem(key);
        if (savedValue) {
            input.value = savedValue;
        }
    });
}

function initializeCharts() {
    // Prevent main.js from initializing charts on the results page
    if (window.location.pathname.includes('/results/')) {
        return;
    }
    // Example chart initialization
    const chartElements = document.querySelectorAll('.chart-container');
    chartElements.forEach(container => {
        const ctx = container.querySelector('canvas');
        if (ctx) {
            // Check if chart already exists and destroy it
            if (ctx.chart) {
                ctx.chart.destroy();
            }
            
            const chartType = container.dataset.chartType || 'line';
            const chartData = JSON.parse(container.dataset.chartData || '{}');
            
            // Create new chart and store reference
            ctx.chart = new Chart(ctx, {
                type: chartType,
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: container.dataset.chartTitle || 'Chart'
                        }
                    }
                }
            });
        }
    });
}

// API helper functions
async function makeAPIRequest(url, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Training functions
async function startTraining(purpose, model, parameters) {
    try {
        const response = await makeAPIRequest(`/api/train/${purpose}/${model}`, 'POST', parameters);
        return response;
    } catch (error) {
        console.error('Training failed:', error);
        throw error;
    }
}

async function getResults(purpose, model) {
    try {
        const response = await makeAPIRequest(`/api/results/${purpose}/${model}`);
        return response;
    } catch (error) {
        console.error('Failed to get results:', error);
        throw error;
    }
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for use in other modules
window.GNNPlatform = {
    makeAPIRequest,
    startTraining,
    getResults,
    showNotification,
    formatNumber,
    debounce
}; 