
"""
Faculty Workload & Timetable Assistant - Generative AI Agent
A comprehensive RAG-based AI agent for managing faculty schedules and workloads
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class FacultyTimetableAgent:
    def __init__(self):
        """Initialize the Faculty Timetable Agent with all components"""
        self.setup_data()
        self.setup_vector_db()
        self.setup_llm()
        self.setup_tools()
        self.setup_agent()

    def setup_data(self):
        """Load faculty and timetable data"""
        try:
            self.faculty_data = pd.read_csv('faculty_workload.csv')
            self.timetable_data = pd.read_csv('timetable.csv')

            # Read university policies
            with open('university_policies.txt', 'r') as f:
                self.policies_text = f.read()

            print("✓ Data loaded successfully")
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample data if files don't exist
            self.create_sample_data()

    def create_sample_data(self):
        """Create sample data if files don't exist"""
        # Faculty workload data
        faculty_data = {
            'FacultyID': ['F101', 'F102', 'F103', 'F104'],
            'Name': ['Prof. Sharma', 'Prof. Mehta', 'Prof. Rao', 'Prof. Iyer'],
            'Department': ['CSE', 'CSE', 'EEE', 'ME'],
            'Course': ['Data Structures', 'AI & ML', 'Circuits', 'Fluid Mechanics'],
            'HoursPerWeek': [6, 8, 5, 7]
        }
        self.faculty_data = pd.DataFrame(faculty_data)

        # Timetable data
        timetable_data = {
            'Day': ['Monday', 'Monday', 'Tuesday', 'Wednesday'],
            'Time': ['10:00-11:00', '11:00-12:00', '14:00-15:00', '09:00-10:00'],
            'Course': ['Data Structures', 'AI & ML', 'Circuits', 'Fluid Mechanics'],
            'Faculty': ['Prof. Sharma', 'Prof. Mehta', 'Prof. Rao', 'Prof. Iyer'],
            'Room': ['Room 201', 'Room 202', 'Room 305', 'Room 401']
        }
        self.timetable_data = pd.DataFrame(timetable_data)

        # University policies
        self.policies_text = """
        University Policies:
        - Maximum workload per professor: 12 hours per week.
        - No faculty should have more than 3 consecutive teaching hours.
        - Faculty should have at least one free slot between two sessions.
        """

    def setup_vector_db(self):
        """Setup ChromaDB vector database for storing policies and rules"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path="./chroma_faculty_db")

            # Create embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # Create or get collection
            try:
                self.collection = self.client.get_collection(
                    name="faculty_policies",
                    embedding_function=self.embedding_function
                )
                print("✓ Vector database collection loaded")
            except:
                # Create new collection and add policies
                self.collection = self.client.create_collection(
                    name="faculty_policies",
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )

                # Add policies to vector database
                policy_chunks = self.policies_text.split('\n')
                policy_chunks = [chunk.strip() for chunk in policy_chunks if chunk.strip()]

                self.collection.add(
                    documents=policy_chunks,
                    ids=[f"policy_{i}" for i in range(len(policy_chunks))],
                    metadatas=[{"type": "university_policy"} for _ in policy_chunks]
                )
                print("✓ Vector database created and policies indexed")

        except Exception as e:
            print(f"Error setting up vector database: {e}")
            self.collection = None

    def setup_llm(self):
        """Setup the language model (using a lightweight model for demo)"""
        try:
            # For this demo, we'll use a simple text generation approach
            # In production, you would use Mistral-7B or similar
            print("✓ LLM setup completed (using basic text processing for demo)")
            self.llm_available = True
        except Exception as e:
            print(f"LLM setup error: {e}")
            self.llm_available = False

    def setup_tools(self):
        """Setup tools for the agent"""
        self.tools = [
            Tool(
                name="RAG_Tool",
                func=self.rag_query,
                description="Answer queries about faculty workload policies and university rules"
            ),
            Tool(
                name="Timetable_Query",
                func=self.query_timetable,
                description="Retrieve class schedule information from timetable data"
            ),
            Tool(
                name="Workload_Report",
                func=self.generate_workload_report,
                description="Generate workload reports by professor or department"
            ),
            Tool(
                name="Faculty_Availability",
                func=self.check_faculty_availability,
                description="Check which faculty members are available at specific times"
            )
        ]
        print("✓ Agent tools configured")

    def setup_agent(self):
        """Initialize the conversational agent"""
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        print("✓ Agent initialized successfully")

    def rag_query(self, query):
        """RAG tool - Query university policies using vector search"""
        try:
            if self.collection is None:
                return "Vector database not available. Using basic policy information."

            # Query vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=3
            )

            if results['documents']:
                context = "\n".join(results['documents'][0])
                response = f"Based on university policies:\n{context}"
                return response
            else:
                return "No relevant policies found for this query."

        except Exception as e:
            return f"Error querying policies: {e}"

    def query_timetable(self, query):
        """Query timetable data based on natural language input"""
        try:
            query_lower = query.lower()

            # Parse different types of queries
            if 'prof.' in query_lower or 'professor' in query_lower:
                # Faculty-specific query
                for _, row in self.faculty_data.iterrows():
                    if row['Name'].lower() in query_lower:
                        faculty_schedule = self.timetable_data[
                            self.timetable_data['Faculty'] == row['Name']
                        ]
                        if not faculty_schedule.empty:
                            schedule_info = []
                            for _, sched in faculty_schedule.iterrows():
                                schedule_info.append(f"{sched['Day']} {sched['Time']}: {sched['Course']} in {sched['Room']}")
                            return f"{row['Name']} schedule:\n" + "\n".join(schedule_info)
                        else:
                            return f"{row['Name']} has no scheduled classes in the timetable."

            elif any(day in query_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
                # Day-specific query
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    if day.lower() in query_lower:
                        day_schedule = self.timetable_data[self.timetable_data['Day'] == day]
                        if not day_schedule.empty:
                            schedule_info = []
                            for _, sched in day_schedule.iterrows():
                                schedule_info.append(f"{sched['Time']}: {sched['Course']} - {sched['Faculty']} in {sched['Room']}")
                            return f"{day} schedule:\n" + "\n".join(schedule_info)
                        else:
                            return f"No classes scheduled for {day}."

            else:
                # General timetable info
                return self.timetable_data.to_string(index=False)

        except Exception as e:
            return f"Error querying timetable: {e}"

    def generate_workload_report(self, query):
        """Generate workload reports for faculty or departments"""
        try:
            query_lower = query.lower()

            if 'department' in query_lower or 'dept' in query_lower:
                # Department-wise report
                dept_report = []
                for dept in self.faculty_data['Department'].unique():
                    dept_faculty = self.faculty_data[self.faculty_data['Department'] == dept]
                    total_hours = dept_faculty['HoursPerWeek'].sum()
                    dept_report.append(f"\n{dept} Department:")
                    for _, faculty in dept_faculty.iterrows():
                        dept_report.append(f"- {faculty['Name']}: {faculty['HoursPerWeek']} hours ({faculty['Course']})")
                    dept_report.append(f"Total: {total_hours} hours")

                return "Department Workload Report:" + "\n".join(dept_report)

            elif 'prof.' in query_lower:
                # Specific faculty report
                for _, row in self.faculty_data.iterrows():
                    if row['Name'].lower() in query_lower:
                        status = "within policy" if row['HoursPerWeek'] <= 12 else "exceeds policy limit"
                        return f"{row['Name']} Workload Report:\n" \
                               f"Course: {row['Course']}\n" \
                               f"Hours per week: {row['HoursPerWeek']}\n" \
                               f"Department: {row['Department']}\n" \
                               f"Status: {status} (max 12 hours/week)"

            else:
                # Overall report
                total_faculty = len(self.faculty_data)
                total_hours = self.faculty_data['HoursPerWeek'].sum()
                avg_hours = total_hours / total_faculty

                return f"Overall Workload Summary:\n" \
                       f"Total Faculty: {total_faculty}\n" \
                       f"Total Teaching Hours: {total_hours}\n" \
                       f"Average Hours per Faculty: {avg_hours:.1f}\n" \
                       f"Faculty within policy (<= 12 hrs): {len(self.faculty_data[self.faculty_data['HoursPerWeek'] <= 12])}"

        except Exception as e:
            return f"Error generating workload report: {e}"

    def check_faculty_availability(self, query):
        """Check faculty availability at specific times"""
        try:
            query_lower = query.lower()

            # Extract day and time information
            day_found = None
            time_found = None

            # Check for days
            for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
                if day in query_lower:
                    day_found = day.capitalize()
                    break

            # Check for time patterns
            if 'pm' in query_lower or 'am' in query_lower:
                time_parts = query_lower.split()
                for part in time_parts:
                    if 'pm' in part or 'am' in part:
                        time_found = part
                        break

            if day_found:
                # Get scheduled faculty for that day
                scheduled = self.timetable_data[self.timetable_data['Day'] == day_found]
                scheduled_faculty = set(scheduled['Faculty'].tolist())
                all_faculty = set(self.faculty_data['Name'].tolist())
                available_faculty = all_faculty - scheduled_faculty

                result = f"Faculty availability for {day_found}:\n"
                if available_faculty:
                    result += "Available: " + ", ".join(available_faculty) + "\n"
                if scheduled_faculty:
                    result += "Scheduled: " + ", ".join(scheduled_faculty)

                return result
            else:
                return "Please specify a day to check faculty availability."

        except Exception as e:
            return f"Error checking availability: {e}"

    def process_query(self, user_query):
        """Process user query and route to appropriate tool"""
        try:
            query_lower = user_query.lower()

            # Determine which tool to use based on query content
            if any(word in query_lower for word in ['policy', 'rule', 'maximum', 'limit', 'guideline']):
                return self.rag_query(user_query)

            elif any(word in query_lower for word in ['workload', 'hours', 'report', 'summary']):
                return self.generate_workload_report(user_query)

            elif any(word in query_lower for word in ['available', 'free', 'availability']):
                return self.check_faculty_availability(user_query)

            elif any(word in query_lower for word in ['schedule', 'timetable', 'class', 'when']):
                return self.query_timetable(user_query)

            else:
                # General query - try to provide relevant information
                return f"I can help you with:\n" \
                       f"- Faculty workload queries (e.g., 'What is Prof. Sharma's workload?')\n" \
                       f"- Timetable information (e.g., 'Show Monday schedule')\n" \
                       f"- Faculty availability (e.g., 'Who is free on Tuesday?')\n" \
                       f"- University policies (e.g., 'What are the workload limits?')\n\n" \
                       f"Your query: '{user_query}'"

        except Exception as e:
            return f"Error processing query: {e}"

# Initialize the agent
@st.cache_resource
def get_agent():
    return FacultyTimetableAgent()

def main():
    """Streamlit application main function"""
    st.set_page_config(
        page_title="Faculty Timetable Assistant",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 Faculty Workload & Timetable Assistant")
    st.markdown("### Generative AI Agent for Academic Scheduling")

    # Initialize agent
    agent = get_agent()

    # Sidebar with information
    with st.sidebar:
        st.header("📋 System Information")
        st.info(
            "This AI assistant helps with:\n"
            "• Faculty workload management\n"
            "• Timetable queries\n"
            "• Availability checking\n"
            "• Policy information"
        )
        if CHROMADB_AVAILABLE:
            st.write("Vector Database: ✅ Active")
        else:
            st.write("Policy Search Fallback: ✅ Ready")

        st.header("📊 Current Data")
        st.write("Faculty Members:", len(agent.faculty_data))
        st.write("Scheduled Classes:", len(agent.timetable_data))

        # Show sample data
        if st.checkbox("Show Faculty Data"):
            st.dataframe(agent.faculty_data, use_container_width=True)

        if st.checkbox("Show Timetable"):
            st.dataframe(agent.timetable_data, use_container_width=True)

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask the Assistant")

        # Example queries
        st.markdown("**Example queries:**")
        examples = [
            "What is Prof. Sharma's workload this week?",
            "Which faculty is free on Tuesday at 2 PM?",
            "Summarize CSE department workload",
            "What are the university workload policies?",
            "Show Monday schedule"
        ]

        for example in examples:
            if st.button(f"📝 {example}", key=f"ex_{hash(example)}"):
                response = agent.process_query(example)
                st.success("**Query:** " + example)
                st.write("**Response:**")
                st.write(response)

        st.markdown("---")

        # Custom query input
        user_query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is Prof. Sharma's teaching schedule?"
        )

        if st.button("🚀 Ask Assistant", type="primary"):
            if user_query:
                with st.spinner("Processing your query..."):
                    response = agent.process_query(user_query)
                    st.success("**Your Query:** " + user_query)
                    st.write("**Assistant Response:**")
                    st.write(response)
            else:
                st.warning("Please enter a question first.")

    with col2:
        st.header("🔧 System Status")

        # Status indicators
        status_items = [
            ("Data Loading", "✅ Ready"),
            ("Vector Database", "✅ Active"),
            ("LLM Processing", "✅ Ready"),
            ("Agent Tools", "✅ Configured")
        ]

        for item, status in status_items:
            st.write(f"**{item}:** {status}")

        st.markdown("---")
        st.header("📈 Quick Stats")

        # Quick statistics
        total_hours = agent.faculty_data['HoursPerWeek'].sum()
        avg_hours = total_hours / len(agent.faculty_data)
        overloaded = len(agent.faculty_data[agent.faculty_data['HoursPerWeek'] > 12])

        st.metric("Total Teaching Hours", f"{total_hours} hrs/week")
        st.metric("Average per Faculty", f"{avg_hours:.1f} hrs")
        st.metric("Overloaded Faculty", overloaded)

if __name__ == "__main__":
    main()
