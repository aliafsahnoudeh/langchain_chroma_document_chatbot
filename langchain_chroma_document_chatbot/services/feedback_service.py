from langsmith import Client


class FeedbackService:
    client: Client

    def __init__(self) -> None:
        self.client = Client()

    def persist(self, run_id: str, feedback_key: str, score: bool) -> str:
        feedback = self.client.create_feedback(
            run_id=run_id,
            key=feedback_key,
            score=score,
        )
        return str(feedback.id)
