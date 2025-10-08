"""
Faculty Workload & Timetable Assistant - Generative AI Agent
A comprehensive RAG-based AI agent for managing faculty schedules and workloads
"""

import pandas as pd
import streamlit as st
from langchain_core.tools import Tool 
from langchain.memory import ConversationBufferMemory
import re
import warnings
warnings.filterwarnings("ignore")

class FacultyTimetableAgent:
    def __init__(self):
        self.setup_data()
        self.setup_vector_db()
        self.setup_tools()
        self.setup_agent()

    def setup_data(self):
        try:
            self.faculty_data = pd.read_csv('faculty_workload.csv')
            self.timetable_data = pd.read_csv('timetable.csv')
            with open('university_policies.txt', 'r') as f:
                self.policies_text = f.read()
            print("✓ Data loaded successfully")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.create_sample_data()

    def create_sample_data(self):
        faculty_data = {
            'FacultyID': ['F101', 'F102', 'F103', 'F104'],
            'Name': ['Prof. Sharma', 'Prof. Mehta', 'Prof. Rao', 'Prof. Iyer'],
            'Department': ['CSE', 'CSE', 'EEE', 'ME'],
            'Course': ['Data Structures', 'AI & ML', 'Circuits', 'Fluid Mechanics'],
            'HoursPerWeek': [6, 8, 5, 7]
        }
        self.faculty_data = pd.DataFrame(faculty_data)

        timetable_data = {
            'Day': ['Monday', 'Monday', 'Tuesday', 'Wednesday'],
            'Time': ['10:00-11:00', '11:00-12:00', '14:00-15:00', '09:00-10:00'],
            'Course': ['Data Structures', 'AI & ML', 'Circuits', 'Fluid Mechanics'],
            'Faculty': ['Prof. Sharma', 'Prof. Mehta', 'Prof. Rao', 'Prof. Iyer'],
            'Room': ['Room 201', 'Room 202', 'Room 305', 'Room 401']
        }
        self.timetable_data = pd.DataFrame(timetable_data)

        self.policies_text = (
            "University Policies:\n"
            "- Maximum workload per professor: 12 hours per week.\n"
            "- No faculty should have more than 3 consecutive teaching hours.\n"
            "- Faculty should have at least one free slot between two sessions.\n"
        )

    def setup_vector_db(self):
        """Try to setup chromadb if available. Otherwise, fallback."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            self.client = chromadb.PersistentClient(path="./chroma_faculty_db")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            try:
                self.collection = self.client.get_collection(
                    name="faculty_policies",
                    embedding_function=self.embedding_function
                )
            except Exception:
                self.collection = self.client.create_collection(
                    name="faculty_policies",
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                policy_chunks = [chunk.strip() for chunk in self.policies_text.split('\n') if chunk.strip()]
                self.collection.add(
                    documents=policy_chunks,
                    ids=[f"policy_{i}" for i in range(len(policy_chunks))],
                    metadatas=[{"type": "university_policy"} for _ in policy_chunks]
                )
            self.vector_search_available = True
        except Exception as e:
            print("chromadb unavailable, using fallback:", e)
            self.collection = None
            self.vector_search_available = False

    def setup_tools(self):
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
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def rag_query(self, query):
        """Query university policies using vector DB (if available) or fallback"""
        if getattr(self, "vector_search_available", False) and self.collection is not None:
            try:
                results = self.collection.query(
                    query_texts=[query], n_results=3
                )
                documents = results.get('documents', [])
                if documents and documents[0]:
                    context = "\n".join(documents[0])
                    return f"Based on university policies:\n{context}"
                else:
                    return "No relevant policies found for this query."
            except Exception as e:
                return f"Error querying policies: {e}"
        else:
            # Fallback: basic keyword search
            keywords = re.findall(r'\w+', query.lower())
            lines = self.policies_text.split('\n')
            relevant_lines = [line for line in lines if any(k in line.lower() for k in keywords)]
            if relevant_lines:
                return "Policy Search Fallback:\n" + "\n".join(relevant_lines)
            else:
                return "No matching policies found. Please refine your query."

    def query_timetable(self, query):
        try:
            query_lower = query.lower()
            if 'prof.' in query_lower or 'professor' in query_lower:
                for _, row in self.faculty_data.iterrows():
                    if row['Name'].lower() in query_lower:
                        faculty_schedule = self.timetable_data[self.timetable_data['Faculty'] == row['Name']]
                        if not faculty_schedule.empty:
                            schedule_info = []
                            for _, sched in faculty_schedule.iterrows():
                                schedule_info.append(f"{sched['Day']} {sched['Time']}: {sched['Course']} in {sched['Room']}")
                            return f"{row['Name']} schedule:\n" + "\n".join(schedule_info)
                        else:
                            return f"{row['Name']} has no scheduled classes in the timetable."
            elif any(day in query_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
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
                return self.timetable_data.to_string(index=False)
        except Exception as e:
            return f"Error querying timetable: {e}"

    def generate_workload_report(self, query):
        try:
            query_lower = query.lower()
            if 'department' in query_lower or 'dept' in query_lower:
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
                for _, row in self.faculty_data.iterrows():
                    if row['Name'].lower() in query_lower:
                        status = "within policy" if row['HoursPerWeek'] <= 12 else "exceeds policy limit"
                        return f"{row['Name']} Workload Report:\n" \
                               f"Course: {row['Course']}\n" \
                               f"Hours per week: {row['HoursPerWeek']}\n" \
                               f"Department: {row['Department']}\n" \
                               f"Status: {status} (max 12 hours/week)"
            else:
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
        try:
            query_lower = query.lower()
            day_found = None
            for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
                if day in query_lower:
                    day_found = day.capitalize()
                    break
            if day_found:
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
        try:
            query_lower = user_query.lower()
            if any(word in query_lower for word in ['policy', 'rule', 'maximum', 'limit', 'guideline']):
                return self.rag_query(user_query)
            elif any(word in query_lower for word in ['workload', 'hours', 'report', 'summary']):
                return self.generate_workload_report(user_query)
            elif any(word in query_lower for word in ['available', 'free', 'availability']):
                return self.check_faculty_availability(user_query)
            elif any(word in query_lower for word in ['schedule', 'timetable', 'class', 'when']):
                return self.query_timetable(user_query)
            else:
                return (
                    f"I can help you with:\n"
                    f"- Faculty workload queries (e.g., 'What is Prof. Sharma's workload?')\n"
                    f"- Timetable information (e.g., 'Show Monday schedule')\n"
                    f"- Faculty availability (e.g., 'Who is free on Tuesday?')\n"
                    f"- University policies (e.g., 'What are the workload limits?')\n\n"
                    f"Your query: '{user_query}'"
                )
        except Exception as e:
            return f"Error processing query: {e}"

@st.cache_resource
def get_agent():
    return FacultyTimetableAgent()

def main():
    st.set_page_config(
        page_title="Faculty Timetable Assistant",
        page_icon="🎓",
        layout="wide"
    )
    st.title("🎓 Faculty Workload & Timetable Assistant")
    st.markdown("### Generative AI Agent for Academic Scheduling")
    agent = get_agent()

    with st.sidebar:
        st.header("📋 System Information")
        st.info(
            "This AI assistant helps with:\n"
            "• Faculty workload management\n"
            "• Timetable queries\n"
            "• Availability checking\n"
            "• Policy information"
        )
        if hasattr(agent, 'vector_search_available') and agent.vector_search_available:
            st.write("Vector Database: ✅ Active")
        else:
            st.write("Policy Search Fallback: ✅ Ready")

        st.header("📊 Current Data")
        st.write("Faculty Members:", len(agent.faculty_data))
        st.write("Scheduled Classes:", len(agent.timetable_data))
        if st.checkbox("Show Faculty Data"):
            st.dataframe(agent.faculty_data, use_container_width=True)
        if st.checkbox("Show Timetable"):
            st.dataframe(agent.timetable_data, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("💬 Ask the Assistant")
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
        status_items = [
            ("Data Loading", "✅ Ready"),
            (
                "Vector Database" if hasattr(agent, 'vector_search_available') and agent.vector_search_available 
                else "Policy Search Fallback",
                "✅ Active" if hasattr(agent, 'vector_search_available') and agent.vector_search_available else "✅ Ready"
            ),
            ("Agent Tools", "✅ Configured")
        ]
        for item, status in status_items:
            st.write(f"**{item}:** {status}")
        st.markdown("---")
        st.header("📈 Quick Stats")
        total_hours = agent.faculty_data['HoursPerWeek'].sum()
        avg_hours = total_hours / len(agent.faculty_data)
        overloaded = len(agent.faculty_data[agent.faculty_data['HoursPerWeek'] > 12])
        st.metric("Total Teaching Hours", f"{total_hours} hrs/week")
        st.metric("Average per Faculty", f"{avg_hours:.1f} hrs")
        st.metric("Overloaded Faculty", overloaded)

if __name__ == "__main__":
    main()
