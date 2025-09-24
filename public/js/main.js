// Main JavaScript file for Money Money website

// Utility functions
function formatPrice(price) {
    return parseFloat(price).toFixed(4);
}

function formatPercentage(percent) {
    return (percent >= 0 ? '+' : '') + percent.toFixed(2) + '%';
}

// Landing page functionality
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
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

    // Newsletter form handling
    const newsletterForm = document.querySelector('.newsletter-form');
    if (newsletterForm) {
        newsletterForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const email = this.querySelector('input[type="email"]').value;

            if (email) {
                // Simulate newsletter signup
                alert('Thank you for subscribing! We\'ll keep you updated with the latest trading signals.');
                this.querySelector('input[type="email"]').value = '';
            }
        });
    }

    // Add scroll effect to navbar
    const header = document.querySelector('.main-header');
    if (header && header.style.position === 'absolute') {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 100) {
                header.style.backgroundColor = 'rgba(12, 11, 16, 0.95)';
                header.style.backdropFilter = 'blur(10px)';
                header.style.borderBottom = '1px solid rgba(255, 255, 255, 0.1)';
            } else {
                header.style.backgroundColor = 'transparent';
                header.style.backdropFilter = 'none';
                header.style.borderBottom = 'none';
            }
        });
    }

    // Animate cards on scroll
    const cards = document.querySelectorAll('.card');
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const cardObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        cardObserver.observe(card);
    });

    // Add typing effect to hero title (if on landing page)
    const heroTitle = document.querySelector('.hero h1');
    if (heroTitle && heroTitle.textContent.includes('Trading')) {
        const text = heroTitle.innerHTML;
        heroTitle.innerHTML = '';
        let index = 0;

        function typeText() {
            if (index < text.length) {
                heroTitle.innerHTML = text.slice(0, index + 1);
                index++;
                setTimeout(typeText, 50);
            }
        }

        setTimeout(typeText, 1000);
    }
});

// Mobile menu toggle (for future enhancement)
function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    if (navLinks) {
        navLinks.classList.toggle('mobile-active');
    }
}

// Theme toggle functionality (for future enhancement)
function toggleTheme() {
    document.body.classList.toggle('light-theme');
    const themeIcon = document.querySelector('.theme-toggle');
    if (themeIcon) {
        themeIcon.textContent = document.body.classList.contains('light-theme') ? '🌙' : '☀️';
    }
}

// Price update animation
function animatePrice(element, newPrice, oldPrice) {
    if (!element) return;

    const isIncrease = parseFloat(newPrice) > parseFloat(oldPrice);
    element.style.color = isIncrease ? '#22c55e' : '#ef4444';
    element.style.transform = 'scale(1.1)';

    setTimeout(() => {
        element.style.transform = 'scale(1)';
        element.style.color = '';
    }, 300);
}

// Error handling for images
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('error', function() {
            // Replace with placeholder if image fails to load
            this.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="%23E1007A"/><text x="50" y="50" text-anchor="middle" dy="0.3em" fill="white" font-family="Arial">No Image</text></svg>';
        });
    });
});

// Copy to clipboard functionality
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Copied to clipboard!');
        });
    } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showNotification('Copied to clipboard!');
    }
}

// Show notification
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#22c55e' : '#ef4444'};
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }

    .mobile-active {
        display: flex !important;
        flex-direction: column;
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: rgba(12, 11, 16, 0.95);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 0 0 16px 16px;
    }

    @media (max-width: 768px) {
        .mobile-active {
            display: flex !important;
        }
    }
`;
document.head.appendChild(style);

// Export functions for global use
window.MoneyMoney = {
    toggleMobileMenu,
    toggleTheme,
    animatePrice,
    copyToClipboard,
    showNotification,
    formatPrice,
    formatPercentage
};