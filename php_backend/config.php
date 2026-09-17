<?php
// Database and mail configuration.
// All secrets come from environment variables (set in Render's
// dashboard for each service). Local defaults are only used when
// the env var isn't set, so this still works unchanged on XAMPP.

function env_or($key, $default) {
    $value = getenv($key);
    return ($value !== false && $value !== '') ? $value : $default;
}

return [
    'db' => [
        'host' => env_or('DB_HOST', '127.0.0.1'),
        'port' => (int) env_or('DB_PORT', 3306),
        'dbname' => env_or('DB_NAME', 'fraud_detection'),
        'username' => env_or('DB_USER', 'root'),
        'password' => env_or('DB_PASSWORD', ''),
        'charset' => 'utf8mb4',
    ],

    // Python model backend URL (used to score transactions before saving).
    // On Render this becomes the FastAPI service's internal URL,
    // e.g. http://smartdetector-api:10000/predict_simple
    'model_api_url' => env_or('MODEL_API_URL', 'http://localhost:8000/predict_simple'),

    'mail' => [
        'from_address' => env_or('MAIL_FROM_ADDRESS', 'noreply@simplepay.local'),
        'from_name' => env_or('MAIL_FROM_NAME', 'SimplePay Alerts'),
        'subject_prefix' => env_or('MAIL_SUBJECT_PREFIX', '[SimplePay] '),
        'smtp' => [
            'enabled' => true,
            'host' => env_or('SMTP_HOST', 'smtp.gmail.com'),
            'port' => (int) env_or('SMTP_PORT', 587),
            'secure' => env_or('SMTP_SECURE', 'tls'),
            'auth' => true,
            'username' => env_or('SMTP_USERNAME', ''),
            'password' => env_or('SMTP_PASSWORD', ''),
        ]
    ],
];
