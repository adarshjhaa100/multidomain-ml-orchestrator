### PURPOSE:

- To evaluate the system based on specific tasks. Tasks should be “random” and not the ones on which the original model might’ve trained on
- Solution should be able to properly plan and decide the level of Question

---

### Small coding task

- small code snippets, easy level DSA questions, maths and introductory programming session
  - fibonacci, prime number basic approach, count words, basic tree traversal, graph traversal
  - Write a program to take two numbers n and k as input, where n >= k and print the total sum of cube of every kth number from 1 to n
  - Write a program to create a hashmap from scratch and add some corner cases to simulate the testing
  - Write an HTML + JS code using canvas 2D to Create a rotating trapezium along centre
  - Write an HTML + JS + CSS code to create a grid view todo list with clean graphics. Users can add/edit/remove todo
  - Create a small math library in pure C for: multiplying a number k times, differentiation of a given polynomial math function, division by float

- Actual Coding questions from here ( They're pretty mixed up, the above is better )
  - https://huggingface.co/datasets/Phonsiri/Qwen3.5-Distillation-Dataset/viewer?row=22

---

### Complex coding task

- Requires planning

- Implement Medium, Hard Level DSA question

- Advent of code

- Implement testable backends from system design questions ( Multi Query )
  - Design and implement a concurrent hashmap in Pure C and write actual benchmark for it to simulate with configurable number of requests where requests are in magnitude of millions. Make sure to utilize all cores. Also, make sure the data auto syncs with disk.
  - Design and implement a data processing and storage pipeline and executor in Pure C. The system should efficiently pack and retrieve data, allow OLAP querying and storage while squeezing last bit of hardware power. Write actual benchmark for it to simulate with configurable number of requests where requests are in magnitude of millions. Make sure to utilize all cores. Also, make sure the data auto syncs with disk.
  - Design and implement a data processing and storage pipeline and executor in Pure C. The system should efficiently pack and retrieve data, allow OLAP querying and storage while squeezing last bit of hardware power. Write actual benchmark for it to simulate with configurable number of requests where requests are in magnitude of millions. Make sure to utilize all cores. Also, make sure the data auto syncs with disk.
  - Let's implement a multiplayer game using vanilla html 2d canvas + javascript + css and connect clients through either of : websocket, polling wichever suits low latency requirements. Goal is to have 100s concurrent clients. It will be a simple RPG with pokemon fire red styled graphics where players are lost on a survival island and gather and compete for resources: food, water, shelter, able to craft weapon ( a single one for now ) and fight. Goal is to survive and roleplay. Add dangerous animals as well.

### Small QnA on GS

- Capitals of cities
- Questions from world general studies
  - What is the shortest way to travel from Washington DC to New York by road or sea
  - What is climate change
  - Who is most responsible for pollution
  - Which element is most abundant on earth's surface


### Complex research prompt

- How to fix climate change
- Please analyze properly based on data: Who are responsible for most climate change based on per capita emissions, lifestyle, business behaviors, etc. List top 10 people in the work or organizations pls.
- future problem statements
- Test evaluation on a random generated quiz
- Recent current affairs
- STEM research questions with a bit of maths and logical analytics
- Questions from https://huggingface.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b/viewer/stage1/train?row=6


### Custom problems:

- Build a database from scratch in C with crud functionality, In memory first approach with WAL and ACID transactions.
- Game simulation for 2d grid in form of matrix
- Design and implement a 3d simulation game using pure html, css and js where user creates a custom grid and game automatically spawns a doom styled game with enemies, traps et.c. Make sure to utilize all cores.
- Design and implement a MapReduce based on tigerstyle and storage pipeline and executor in Pure C. The system should efficiently pack and retrieve and process data while squeezing last bit of hardware power. Write actual benchmark for it to simulate with configurable number of requests where requests are in magnitude of millions. Make sure to utilize all cores. Also, make sure the data auto syncs with disk.
- Using data from openstreetmaps, design a 3d navigator in pure HTML + CSS + JS, plan routes, plan detailed trips on a side component which will pop up when clicked "plan trip" where users would be able to choose hotels, plan treks, food, fun activities. Add functionality to mark basic amenities in hexagonal region of 1KM square and also mark a safety score.
- Questions from https://huggingface.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b/viewer/stage1/train?row=6
