/**
 * KLSBench Interactive Components
 */

// Global state
let currentTask = 'classification';
let samplesData = {};
let currentSampleIndex = 0;

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 KLSBench Data Explorer initialized');

    // Load summary data
    await loadSummary();

    // Load all sample data
    await loadAllSamples();

    // Set up event listeners
    setupEventListeners();

    // Display initial samples
    displaySamples(currentTask);
});

/**
 * Load summary statistics
 */
async function loadSummary() {
    try {
        const response = await fetch('samples/summary.json');
        const data = await response.json();

        // Update total instances
        const totalElement = document.getElementById('total-instances');
        if (totalElement) {
            totalElement.textContent = data.total_instances.toLocaleString();
        }

        console.log('📊 Summary loaded:', data);
    } catch (error) {
        console.error('Error loading summary:', error);
    }
}

/**
 * Load all sample data
 */
async function loadAllSamples() {
    const tasks = ['classification', 'retrieval', 'nli', 'translation', 'punctuation'];

    for (const task of tasks) {
        try {
            const response = await fetch(`samples/sample_${task}.json`);
            const data = await response.json();
            samplesData[task] = data;
            console.log(`✅ ${task} samples loaded: ${data.sample_size} instances`);
        } catch (error) {
            console.error(`Error loading ${task} samples:`, error);
        }
    }
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Task selector buttons
    const taskButtons = document.querySelectorAll('.task-btn');
    taskButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const task = e.target.dataset.task;
            selectTask(task);
        });
    });

    // Random sample button
    const randomBtn = document.getElementById('random-sample-btn');
    if (randomBtn) {
        randomBtn.addEventListener('click', () => {
            showRandomSample(currentTask);
        });
    }

    // Task cards
    const taskCards = document.querySelectorAll('.task-card');
    taskCards.forEach(card => {
        card.addEventListener('click', (e) => {
            const task = e.currentTarget.dataset.task;
            if (task) {
                selectTask(task);
                // Smooth scroll to explorer
                document.getElementById('explorer').scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

/**
 * Select a task
 */
function selectTask(task) {
    currentTask = task;
    currentSampleIndex = 0;

    // Update button states
    document.querySelectorAll('.task-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.task === task) {
            btn.classList.add('active');
        }
    });

    // Display samples
    displaySamples(task);
}

/**
 * Display samples for a task
 */
function displaySamples(task) {
    const displayElement = document.getElementById('sample-display');
    const countElement = document.getElementById('sample-count');

    if (!samplesData[task]) {
        displayElement.innerHTML = '<div class="loading">Loading samples...</div>';
        return;
    }

    const data = samplesData[task];

    // Update count
    if (countElement) {
        countElement.textContent = `Showing ${data.sample_size} of ${data.original_size} instances`;
    }

    // Render samples based on task type
    let html = '';

    switch (task) {
        case 'classification':
            html = renderClassificationSamples(data.data);
            break;
        case 'retrieval':
            html = renderRetrievalSamples(data.data);
            break;
        case 'nli':
            html = renderNLISamples(data.data);
            break;
        case 'translation':
            html = renderTranslationSamples(data.data);
            break;
        case 'punctuation':
            html = renderPunctuationSamples(data.data);
            break;
        default:
            html = '<div class="loading">Unknown task type</div>';
    }

    displayElement.innerHTML = html;
}

/**
 * Show a random sample
 */
function showRandomSample(task) {
    if (!samplesData[task] || !samplesData[task].data.length) return;

    const data = samplesData[task].data;
    const randomIndex = Math.floor(Math.random() * data.length);
    const sample = data[randomIndex];

    // Scroll to top of sample display
    const displayElement = document.getElementById('sample-display');
    displayElement.scrollIntoView({ behavior: 'smooth' });

    // Highlight the random sample
    setTimeout(() => {
        const sampleElements = displayElement.querySelectorAll('.sample-item');
        if (sampleElements[randomIndex]) {
            sampleElements[randomIndex].style.border = '2px solid #3498db';
            sampleElements[randomIndex].style.backgroundColor = '#f0f8ff';

            // Scroll to the specific sample
            sampleElements[randomIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });

            // Reset after 3 seconds
            setTimeout(() => {
                sampleElements[randomIndex].style.border = '';
                sampleElements[randomIndex].style.backgroundColor = '';
            }, 3000);
        }
    }, 300);
}

/**
 * Render classification samples
 */
function renderClassificationSamples(samples) {
    return samples.map(sample => `
        <div class="sample-item">
            <div class="sample-id">ID: ${sample.id}</div>
            <div class="sample-content">
                <div class="sample-label">Input (문장):</div>
                <div class="sample-text classical-text">${sample.input}</div>
            </div>
            <div class="sample-answer">
                <strong>Label:</strong> ${sample.label}
            </div>
        </div>
    `).join('');
}

/**
 * Render retrieval samples
 */
function renderRetrievalSamples(samples) {
    return samples.map(sample => `
        <div class="sample-item">
            <div class="sample-id">ID: ${sample.id}</div>
            <div class="sample-content">
                <div class="sample-label">Input (문장):</div>
                <div class="sample-text classical-text">${sample.input}</div>
            </div>
            <div class="sample-answer">
                <strong>Source:</strong> ${sample.answer}<br>
                <strong>Book:</strong> ${sample.book}<br>
                <strong>Chapter:</strong> ${sample.chapter}
            </div>
        </div>
    `).join('');
}

/**
 * Render NLI samples
 */
function renderNLISamples(samples) {
    const labelColors = {
        'entailment': '#4caf50',
        'contradiction': '#f44336',
        'neutral': '#ff9800'
    };

    return samples.map(sample => `
        <div class="sample-item">
            <div class="sample-id">ID: ${sample.id} | Difficulty: ${sample.difficulty} | Category: ${sample.category}</div>
            <div class="sample-content">
                <div class="sample-label">Premise (전제):</div>
                <div class="sample-text">${sample.premise}</div>
            </div>
            <div class="sample-content">
                <div class="sample-label">Hypothesis (가설):</div>
                <div class="sample-text">${sample.hypothesis}</div>
            </div>
            <div class="sample-answer" style="border-left-color: ${labelColors[sample.label]}">
                <strong>Label:</strong> ${sample.label}<br>
                <strong>Source:</strong> ${sample.source}<br>
                ${sample.explanation ? `<strong>Explanation:</strong> ${sample.explanation}` : ''}
            </div>
        </div>
    `).join('');
}

/**
 * Render translation samples
 */
function renderTranslationSamples(samples) {
    return samples.map(sample => `
        <div class="sample-item">
            <div class="sample-id">ID: ${sample.id} | ${sample.source_lang} → ${sample.target_lang}</div>
            <div class="sample-content">
                <div class="sample-label">Source Text (원문):</div>
                <div class="sample-text classical-text">${sample.source_text}</div>
            </div>
            <div class="sample-answer">
                <strong>Translation (번역):</strong><br>
                ${sample.target_text}
                ${sample.book ? `<br><br><strong>Book:</strong> ${sample.book}` : ''}
            </div>
        </div>
    `).join('');
}

/**
 * Render punctuation samples
 */
function renderPunctuationSamples(samples) {
    return samples.map(sample => `
        <div class="sample-item">
            <div class="sample-id">ID: ${sample.id} | ${sample.text_type}</div>
            <div class="sample-content">
                <div class="sample-label">Input (백문):</div>
                <div class="sample-text classical-text">${sample.input}</div>
            </div>
            <div class="sample-answer">
                <strong>Answer (구두점 복원):</strong><br>
                <div class="classical-text">${sample.answer}</div>
                ${sample.metadata && sample.metadata.korean_translation ?
                    `<br><br><strong>Korean Translation:</strong><br>${sample.metadata.korean_translation}` : ''}
            </div>
        </div>
    `).join('');
}

/**
 * Utility: Format numbers
 */
function formatNumber(num) {
    return num.toLocaleString();
}

/**
 * Utility: Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add scroll animation observer
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe task cards and result cards
window.addEventListener('load', () => {
    document.querySelectorAll('.task-card, .result-card, .highlight-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

console.log('✨ KLSBench scripts loaded successfully');
