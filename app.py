"""
Test script to verify the heart disease prediction system works correctly
Run this after training the model and before deployment
"""

import requests
import json
import os


def test_local_server():
    """Test the Flask app running on localhost"""

    print("=" * 60)
    print("TESTING HEART DISEASE PREDICTION SYSTEM")
    print("=" * 60)

    # Check if model files exist
    print("\n1. Checking model files...")
    files_to_check = [
        'heart_disease_model.pkl',
        'scaler.pkl',
        'feature_columns.pkl'
    ]

    all_files_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"   ✓ {file} found")
        else:
            print(f"   ✗ {file} MISSING!")
            all_files_exist = False

    if not all_files_exist:
        print("\n⚠️  Please run 'python train_model.py' first to generate model files.")
        return

    # Test data - Low risk patient
    print("\n2. Testing with LOW RISK patient data...")
    low_risk_data = {
        'age': 45,
        'sex': 1,
        'cp': 3,
        'trestbps': 120,
        'chol': 200,
        'fbs': 0,
        'restecg': 0,
        'thalach': 160,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 0,
        'ca': 0,
        'thal': 0
    }

    try:
        response = requests.post(
            'http://localhost:5000/predict',
            json=low_risk_data,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Prediction successful!")
            print(f"   - Prediction: {'Disease' if result['prediction'] == 1 else 'No Disease'}")
            print(f"   - Risk Level: {result['risk_level']}")
            print(f"   - No Disease Probability: {result['probability']['no_disease']:.2%}")
            print(f"   - Disease Probability: {result['probability']['disease']:.2%}")
        else:
            print(f"   ✗ Error: Status code {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("   ✗ Cannot connect to server!")
        print("   Please make sure Flask app is running: python app.py")
        return
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return

    # Test data - High risk patient
    print("\n3. Testing with HIGH RISK patient data...")
    high_risk_data = {
        'age': 65,
        'sex': 1,
        'cp': 0,
        'trestbps': 160,
        'chol': 300,
        'fbs': 1,
        'restecg': 1,
        'thalach': 110,
        'exang': 1,
        'oldpeak': 2.5,
        'slope': 2,
        'ca': 3,
        'thal': 2
    }

    try:
        response = requests.post(
            'http://localhost:5000/predict',
            json=high_risk_data,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Prediction successful!")
            print(f"   - Prediction: {'Disease' if result['prediction'] == 1 else 'No Disease'}")
            print(f"   - Risk Level: {result['risk_level']}")
            print(f"   - No Disease Probability: {result['probability']['no_disease']:.2%}")
            print(f"   - Disease Probability: {result['probability']['disease']:.2%}")
        else:
            print(f"   ✗ Error: Status code {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return

    # Test health endpoint
    print("\n4. Testing health check endpoint...")
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            print(f"   ✓ Health check passed: {response.json()}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")

    # Test home page
    print("\n5. Testing home page...")
    try:
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print(f"   ✓ Home page loads successfully")
            if 'Heart Disease Prediction' in response.text:
                print(f"   ✓ Page content looks correct")
            else:
                print(f"   ⚠️  Page content may be incorrect")
        else:
            print(f"   ✗ Error loading home page: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\n✅ If all tests passed, your app is ready for deployment!")
    print("📝 Next step: Push to GitHub and deploy on Render")
    print("\n" + "=" * 60)


def test_deployed_app(url):
    """Test a deployed app on Render or other hosting"""

    print(f"\nTesting deployed app at: {url}")
    print("=" * 60)

    # Test health endpoint
    print("\n1. Testing health check...")
    try:
        response = requests.get(f'{url}/health', timeout=30)
        if response.status_code == 200:
            print(f"   ✓ Health check passed")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return

    # Test prediction
    print("\n2. Testing prediction...")
    test_data = {
        'age': 50,
        'sex': 1,
        'cp': 2,
        'trestbps': 130,
        'chol': 250,
        'fbs': 0,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 1.0,
        'slope': 1,
        'ca': 1,
        'thal': 1
    }

    try:
        response = requests.post(
            f'{url}/predict',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Prediction successful!")
            print(f"   - Risk Level: {result['risk_level']}")
        else:
            print(f"   ✗ Prediction failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test deployed app
        deployed_url = sys.argv[1]
        test_deployed_app(deployed_url)
    else:
        # Test local app
        print("\n🧪 Testing local Flask application...")
        print("Make sure your Flask app is running (python app.py)")
        input("\nPress Enter when ready to test...")
        test_local_server()

        print("\n\n💡 To test a deployed app, run:")
        print("   python test_app.py https://your-app.onrender.com")