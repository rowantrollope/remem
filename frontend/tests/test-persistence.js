// Simple test to verify localStorage persistence functionality
// This can be run in the browser console to test the implementation

console.log('Testing localStorage persistence functionality...');

// Test data
const testConversation = {
    id: 'test-conv-1',
    question: 'What is the weather like?',
    answer: 'It is sunny today.',
    created_at: new Date().toISOString(),
    confidence: 'high'
};

const testMemorySave = {
    success: true,
    response: {
        memory_id: 'test-memory-1',
        message: 'Memory saved successfully'
    },
    originalText: 'I had a great lunch today',
    timestamp: new Date().toISOString()
};

const testRecallResponse = {
    success: true,
    response: {
        success: true,
        query: 'Tell me about my food experiences',
        mental_state: 'The user has positive food experiences including a great lunch.',
        memories: [],
        memory_count: 1
    },
    originalQuery: 'Tell me about my food experiences',
    timestamp: new Date().toISOString()
};

// Test localStorage operations
const CHAT_STORAGE_KEY = 'memory-chat-history';
const STORAGE_VERSION = '1.0';

// Create test data structure
const testData = {
    version: STORAGE_VERSION,
    data: {
        conversations: [testConversation],
        memorySaveResponses: [testMemorySave],
        recallResponses: [testRecallResponse],
        lastUpdated: new Date().toISOString()
    },
    savedAt: new Date().toISOString()
};

// Test saving to localStorage
try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(testData));
    console.log('✓ Successfully saved test data to localStorage');
} catch (error) {
    console.error('✗ Failed to save test data to localStorage:', error);
}

// Test loading from localStorage
try {
    const savedData = localStorage.getItem(CHAT_STORAGE_KEY);
    if (savedData) {
        const parsed = JSON.parse(savedData);
        console.log('✓ Successfully loaded test data from localStorage');
        console.log('Data structure:', {
            version: parsed.version,
            conversations: parsed.data?.conversations?.length || 0,
            memorySaves: parsed.data?.memorySaveResponses?.length || 0,
            recallResponses: parsed.data?.recallResponses?.length || 0
        });
    } else {
        console.log('✗ No data found in localStorage');
    }
} catch (error) {
    console.error('✗ Failed to load test data from localStorage:', error);
}

// Test clearing localStorage
try {
    localStorage.removeItem(CHAT_STORAGE_KEY);
    console.log('✓ Successfully cleared test data from localStorage');
} catch (error) {
    console.error('✗ Failed to clear test data from localStorage:', error);
}

console.log('Persistence test completed. Check the console output above for results.');
