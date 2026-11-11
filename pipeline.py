from agents import critic_chain, writer_chain, reader_agent, search_agent

def run_research_pipeline(topic:str) -> dict:
    
    state = {}

    #Search Agent Working

    print("\n"+" ="*50)
    print("Step One - Search Agent Is Working...")
    print("="*50)

    find_agent = search_agent()
    search_result = find_agent.invoke({
        'messages' : [("user",f"Find recent, reliable and information about: {topic}")]
    })
    state['search_result'] = search_result['messages'][-1].content

    print("\n search results",state['search_result'])

# Step 2 Reader Agent Working

    print("\n"+" ="*50)
    print("Step One - Reader Agent Is Working...")
    print("="*50)

    read_agent = reader_agent()
    reader_results = read_agent.invoke({
        'messages' : [("user",
            f"Based On the following search result about {topic}"
            f"Pick The Most Relevant URL and Scrap it for deeper content.\n\n"
            f"Search results: \n {state['search_result'][:800]}"    
        )]
    })
    state['scraped_content']=reader_results['messages'][-1].content

    print("\nScraped Content\n", state['scraped_content'])

#Step - 3 Writer Chain

    print("\n"+" ="*50)
    print("Step 3 Writer Chain Is Working...")
    print("="*50)


    research_combined = (
        f"SEARCH RESULTS : \n {state['search_result']}\n\n"
        f"DETAILED SCRAPED RESULTS : \n {state['scraped_content']}"
    )
    state['report'] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])

    #critic report

    print("\n"+" ="*50)
    print("Step 4 Critic Chain Is Working On Report...")
    print("="*50)

    state['feedback'] = critic_chain.invoke({
        "report" : state['report']
    })

    print("\n Critic report \n",state['feedback'])

    return state

if __name__ == "__main__":
    topic = str(input("\n Enter A Research Topic: "))
    run_research_pipeline(topic)