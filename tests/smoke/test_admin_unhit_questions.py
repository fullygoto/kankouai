from app import app


def test_admin_unhit_questions():
    app.config["TESTING"] = True
    with app.test_client() as client:
        with client.session_transaction() as session:
            session["user_id"] = "admin"
            session["role"] = "admin"

        response = client.get("/admin/unhit_questions")
        assert response.status_code == 200
