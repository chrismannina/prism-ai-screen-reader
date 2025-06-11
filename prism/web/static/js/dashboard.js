// Prism Dashboard JavaScript
class PrismDashboard {
    constructor() {
        this.socket = io();
        this.currentTab = 'screenshots';
        this.activityChart = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.setupSocketEvents();
    }
    
    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab || e.target.closest('.tab-btn').dataset.tab;
                this.switchTab(tabName);
            });
        });
        
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadInitialData();
        });
        
        // Screenshot limit selector
        document.getElementById('screenshot-limit').addEventListener('change', () => {
            this.loadScreenshots();
        });
        
        // Activity hours selector
        document.getElementById('activity-hours').addEventListener('change', () => {
            this.loadActivities();
        });
        
        // Search functionality
        document.getElementById('search-btn').addEventListener('click', () => {
            this.performSearch();
        });
        
        document.getElementById('search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });
        
        // Modal close
        document.getElementById('close-modal').addEventListener('click', () => {
            this.closeModal();
        });
        
        document.getElementById('screenshot-modal').addEventListener('click', (e) => {
            if (e.target.id === 'screenshot-modal') {
                this.closeModal();
            }
        });
    }
    
    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to Prism dashboard');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from Prism dashboard');
            this.updateConnectionStatus(false);
        });
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('status-indicator');
        const dot = indicator.querySelector('div');
        const text = indicator.querySelector('span');
        
        if (connected) {
            dot.className = 'w-3 h-3 bg-green-500 rounded-full animate-pulse';
            text.textContent = 'Connected';
        } else {
            dot.className = 'w-3 h-3 bg-red-500 rounded-full';
            text.textContent = 'Disconnected';
        }
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active', 'border-blue-500', 'text-blue-600');
            btn.classList.add('border-transparent', 'text-gray-600');
        });
        
        const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
        activeBtn.classList.add('active', 'border-blue-500', 'text-blue-600');
        activeBtn.classList.remove('border-transparent', 'text-gray-600');
        
        // Show/hide tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        
        document.getElementById(`tab-${tabName}`).classList.remove('hidden');
        this.currentTab = tabName;
        
        // Load tab-specific data
        this.loadTabData(tabName);
    }
    
    loadTabData(tabName) {
        switch (tabName) {
            case 'screenshots':
                this.loadScreenshots();
                break;
            case 'activities':
                this.loadActivities();
                break;
            case 'search':
                // Search tab doesn't need initial data
                break;
            case 'analytics':
                this.loadAnalytics();
                break;
        }
    }
    
    async loadInitialData() {
        try {
            await this.loadStatus();
            this.loadTabData(this.currentTab);
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showError('Failed to load dashboard data');
        }
    }
    
    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.updateStats(result.data);
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error loading status:', error);
            this.showError('Failed to load status');
        }
    }
    
    updateStats(data) {
        const db = data.database || {};
        const security = data.security || {};
        
        document.getElementById('stats-screenshots').textContent = (db.screenshots_count || 0).toLocaleString();
        document.getElementById('stats-activities').textContent = (db.activities_count || 0).toLocaleString();
        document.getElementById('stats-db-size').textContent = `${(db.database_size_mb || 0).toFixed(1)} MB`;
        document.getElementById('stats-security').textContent = security.encryption_enabled ? 'Enabled' : 'Disabled';
    }
    
    async loadScreenshots() {
        try {
            const limit = document.getElementById('screenshot-limit').value;
            const response = await fetch(`/api/screenshots?limit=${limit}`);
            const result = await response.json();
            
            if (result.status === 'success') {
                this.renderScreenshots(result.data);
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error loading screenshots:', error);
            this.showError('Failed to load screenshots');
        }
    }
    
    renderScreenshots(screenshots) {
        const grid = document.getElementById('screenshots-grid');
        
        if (screenshots.length === 0) {
            grid.innerHTML = '<p class="text-gray-500 text-center col-span-full">No screenshots found</p>';
            return;
        }
        
        grid.innerHTML = screenshots.map(screenshot => `
            <div class="bg-gray-50 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer screenshot-card" 
                 data-id="${screenshot.id}">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-sm text-gray-600">${this.formatDateTime(screenshot.timestamp)}</span>
                    <div class="flex items-center space-x-2">
                        ${screenshot.is_encrypted ? '<i class="fas fa-lock text-green-500"></i>' : '<i class="fas fa-unlock text-gray-400"></i>'}
                        ${screenshot.has_ocr ? '<i class="fas fa-file-text text-blue-500"></i>' : ''}
                    </div>
                </div>
                <div class="mb-3">
                    <div class="bg-gray-200 h-32 rounded-md flex items-center justify-center text-gray-500">
                        <i class="fas fa-image text-2xl"></i>
                        <span class="ml-2">Click to view</span>
                    </div>
                </div>
                <div class="text-xs text-gray-500 space-y-1">
                    <div>Resolution: ${screenshot.resolution}</div>
                    <div>Size: ${screenshot.size_kb} KB</div>
                    ${screenshot.ocr_preview ? `<div class="mt-2 p-2 bg-blue-50 rounded text-blue-800">
                        <i class="fas fa-quote-left text-xs"></i> ${screenshot.ocr_preview}
                    </div>` : ''}
                </div>
            </div>
        `).join('');
        
        // Add click handlers for screenshots
        document.querySelectorAll('.screenshot-card').forEach(card => {
            card.addEventListener('click', () => {
                const screenshotId = card.dataset.id;
                this.viewScreenshot(screenshotId);
            });
        });
    }
    
    async viewScreenshot(screenshotId) {
        try {
            const response = await fetch(`/api/screenshot/${screenshotId}`);
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showScreenshotModal(result.data);
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error viewing screenshot:', error);
            this.showError('Failed to load screenshot');
        }
    }
    
    showScreenshotModal(data) {
        const modalContent = document.getElementById('modal-content');
        modalContent.innerHTML = `
            <div class="space-y-4">
                <div class="text-sm text-gray-600">
                    <strong>Timestamp:</strong> ${this.formatDateTime(data.timestamp)}<br>
                    <strong>Resolution:</strong> ${data.resolution}
                </div>
                <div class="flex justify-center">
                    <img src="${data.image}" alt="Screenshot" class="max-w-full max-h-96 rounded-lg shadow-md">
                </div>
            </div>
        `;
        
        document.getElementById('screenshot-modal').classList.remove('hidden');
    }
    
    closeModal() {
        document.getElementById('screenshot-modal').classList.add('hidden');
    }
    
    async loadActivities() {
        try {
            const hours = document.getElementById('activity-hours').value;
            const response = await fetch(`/api/activities?hours=${hours}&limit=50`);
            const result = await response.json();
            
            if (result.status === 'success') {
                this.renderActivities(result.data);
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error loading activities:', error);
            this.showError('Failed to load activities');
        }
    }
    
    renderActivities(activities) {
        const list = document.getElementById('activities-list');
        
        if (activities.length === 0) {
            list.innerHTML = '<p class="text-gray-500 text-center">No activities found</p>';
            return;
        }
        
        list.innerHTML = activities.map(activity => `
            <div class="bg-gray-50 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-3">
                        <div class="w-3 h-3 rounded-full ${this.getConfidenceColor(activity.confidence)}"></div>
                        <span class="font-medium text-gray-800">${activity.activity_type}</span>
                        <span class="text-sm text-gray-500">${(activity.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <span class="text-xs text-gray-500">${this.formatDateTime(activity.timestamp)}</span>
                </div>
                ${activity.duration_seconds ? `<div class="text-sm text-gray-600">Duration: ${activity.duration_seconds}s</div>` : ''}
            </div>
        `).join('');
    }
    
    getConfidenceColor(confidence) {
        if (confidence > 0.7) return 'bg-green-500';
        if (confidence > 0.5) return 'bg-yellow-500';
        return 'bg-red-500';
    }
    
    async performSearch() {
        const query = document.getElementById('search-input').value.trim();
        if (!query) return;
        
        try {
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=20`);
            const result = await response.json();
            
            if (result.status === 'success') {
                this.renderSearchResults(result.data, query);
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error searching:', error);
            this.showError('Search failed');
        }
    }
    
    renderSearchResults(results, query) {
        const container = document.getElementById('search-results');
        
        if (results.length === 0) {
            container.innerHTML = `<p class="text-gray-500 text-center">No results found for "${query}"</p>`;
            return;
        }
        
        container.innerHTML = `
            <div class="mb-4 text-sm text-gray-600">
                Found ${results.length} result${results.length === 1 ? '' : 's'} for "${query}"
            </div>
            ${results.map(result => `
                <div class="bg-gray-50 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer" 
                     onclick="dashboard.viewScreenshot(${result.id})">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-sm text-gray-600">${this.formatDateTime(result.timestamp)}</span>
                        <span class="text-xs text-gray-500">${result.resolution}</span>
                    </div>
                    <div class="text-sm text-gray-800 bg-yellow-100 p-2 rounded">
                        ${this.highlightSearchTerm(result.context, query)}
                    </div>
                </div>
            `).join('')}
        `;
    }
    
    highlightSearchTerm(text, query) {
        const regex = new RegExp(`(${query})`, 'gi');
        return text.replace(regex, '<mark class="bg-yellow-300">$1</mark>');
    }
    
    async loadAnalytics() {
        try {
            const response = await fetch('/api/summary');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.renderAnalytics(result.data);
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('Error loading analytics:', error);
            this.showError('Failed to load analytics');
        }
    }
    
    renderAnalytics(data) {
        this.renderActivityChart(data);
        this.renderDailySummary(data);
    }
    
    renderActivityChart(data) {
        const ctx = document.getElementById('activity-chart').getContext('2d');
        
        if (this.activityChart) {
            this.activityChart.destroy();
        }
        
        const breakdown = data.activity_breakdown || {};
        const labels = Object.keys(breakdown);
        const values = labels.map(label => breakdown[label].duration || 0);
        
        this.activityChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                datasets: [{
                    data: values,
                    backgroundColor: [
                        '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
                        '#06B6D4', '#84CC16', '#F97316', '#EC4899'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    renderDailySummary(data) {
        const container = document.getElementById('daily-summary');
        const totalMinutes = Math.floor((data.total_duration_seconds || 0) / 60);
        
        container.innerHTML = `
            <div class="space-y-4">
                <div class="bg-blue-50 p-4 rounded-lg">
                    <div class="text-2xl font-bold text-blue-800">${totalMinutes}</div>
                    <div class="text-sm text-blue-600">Total Active Minutes</div>
                </div>
                <div class="bg-green-50 p-4 rounded-lg">
                    <div class="text-2xl font-bold text-green-800">${data.total_activities || 0}</div>
                    <div class="text-sm text-green-600">Total Activities</div>
                </div>
                ${data.activity_breakdown ? `
                    <div class="space-y-2">
                        <h4 class="font-medium text-gray-700">Activity Breakdown:</h4>
                        ${Object.entries(data.activity_breakdown).map(([type, info]) => `
                            <div class="flex justify-between items-center text-sm">
                                <span class="capitalize">${type}</span>
                                <span class="text-gray-600">${Math.floor(info.duration / 60)}m (${info.count} sessions)</span>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    formatDateTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString();
    }
    
    showError(message) {
        // Simple error display - could be enhanced with a toast system
        console.error(message);
        alert(message);
    }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new PrismDashboard();
}); 