import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from roles import PromptEngineer
from datasets import load_dataset
from utils import make_PE_feedback, test_prompt_on_benchmark_async, load_questions
import json
import datetime

app = FastAPI()

# Add CORS middleware to allow cross-origin requests (important for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the dataset
dataset = load_questions()

# Initialize PromptEngineer

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:

            # Wait for a message from the client
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)

            print("json from front end: ", message)


            config = message['config']
            prompt_engineer = PromptEngineer(
                prompt_number=config["prompt_number"],
                parent_model=message["teacher_model"],
                benchmark_name=message["benchmark_name"],
                system_prompt_template=message['system_prompt_template']
            )

            # Check if the message is to start the main loop
            if message.get('action') == "start_main_loop":
                # Run the main loop
                for i in range(config["prompt_number"]):
                    print('===================================')

                    # Generate next prompt
                    next_prompt = prompt_engineer.generate_next_prompt()
                    
                    if i > 1: # don't broadcast the first prompt, it does not contain a score and will break the graph
                        # this is a pretty hacky fix tbh. Should likely be fixed on the frontend
                        await manager.broadcast(json.dumps(prompt_engineer.system_state))
                    
                    # Test the prompt
                    # NOTE: invalid is no longer used in the feedback
                    score, correct, wrong, invalid = await test_prompt_on_benchmark_async(
                        next_prompt,
                        dataset,
                        num_samples=config["num_benchmark_samples"],
                        dataset_type='logicqa2.0',
                        student_model=message["student_model"]
                    )
                    
                    # Generate feedback
                    feedback = make_PE_feedback(
                        score,
                        wrong,
                        num_wrongly_answered=config["num_wrong_feedback_questions"],
                        invalid_answer_decimals=invalid   # this arg no longer used
                    )
                    
                    # Add feedback to PromptEngineer
                    prompt_engineer.add_user_feedback_response(
                        wrongly_answered_qs=wrong,
                        correctly_answered_qs=correct,
                        user_feedback=feedback,
                        score=score
                    )
                    
                    # Update and broadcast system state again
                    await manager.broadcast(json.dumps(prompt_engineer.system_state))

                    # Small delay to prevent flooding
                    await asyncio.sleep(0.1)
                
                # Notify the client that the loop is complete
                await websocket.send_text("main_loop_complete")

                # save to run/{current_time}.json
                # TODO: add saving to a method in PromptEngineer
                now = datetime.datetime.now()
                current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
                with open(f"runs/{current_time}.json", "w") as f:
                    json.dump(prompt_engineer.system_state, f)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

