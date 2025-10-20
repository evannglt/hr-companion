from rag import graph

def main():
    result = graph.invoke({"question": "What is Task Decomposition?"})
    print(result)


if __name__ == "__main__":
    main()
