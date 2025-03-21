"""XML schema definitions for the unified agent module."""

# Input schema - what the agent receives
INPUT_XML_SCHEMA = """
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="agent_state">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="memory" type="xs:string" />
        <xs:element name="last_plan" type="xs:string" />
        <xs:element name="last_action" type="xs:string" />
        <xs:element name="observation" type="xs:string" />
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""

# Output schema - what the agent produces
OUTPUT_XML_SCHEMA = """
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="agent_output">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="updated_memory" type="xs:string" />
        <xs:element name="new_plan" type="xs:string" />
        <xs:element name="execution_instructions" type="xs:string" />
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""

# Example input XML
EXAMPLE_INPUT_XML = """
<agent_state>
  <memory>Previous knowledge about the task</memory>
  <last_plan><action>previous_action</action></last_plan>
  <last_action>command that was executed</last_action>
  <observation>Result of the last action or new information</observation>
</agent_state>
"""

# Example output XML
EXAMPLE_OUTPUT_XML = """
<agent_output>
  <updated_memory>Updated knowledge including new observations</updated_memory>
  <new_plan><action>next_action_to_take</action></new_plan>
  <execution_instructions>Shell command or instructions to execute</execution_instructions>
</agent_output>
"""
