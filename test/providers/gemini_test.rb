# frozen_string_literal: true

require "test_helper"

class GeminiTest < Minitest::Test
  class Transport
    attr_reader :arguments

    def stream(**arguments)
      @arguments = arguments
      chunks = [
        {modelVersion: "gemini", candidates: [{content: {parts: [{text: "Hello"}]}}]},
        {candidates: [{content: {parts: [{functionCall: {id: "tool-1", name: "lookup", args: {id: 1}}}]}, finishReason: "STOP"}],
         usageMetadata: {promptTokenCount: 8, candidatesTokenCount: 5, cachedContentTokenCount: 2, thoughtsTokenCount: 1}}
      ]
      chunks.each { |chunk| yield "data: #{JSON.generate(chunk)}\n\n" }
    end
  end

  def test_streams_text_tools_and_normalized_usage
    transport = Transport.new
    provider = LittleGhost::Providers::Gemini.new(api_key: "secret", model: "gemini", transport:)
    request = LittleGhost::ModelRequest.new(messages: [LittleGhost::Message.new(role: :user, content: "Hi")],
      output_schema: {name: "answer", schema: {type: "object"}})

    events = provider.stream(request).to_a

    assert_equal %i[message_start text_delta tool_call_start tool_call_stop usage message_stop], events.map(&:type)
    response = events.last.data.fetch(:response)
    assert_equal :tool_use, response.stop_reason
    assert_equal 6, response.usage.input_tokens
    assert_equal 4, response.usage.output_tokens
    assert_includes transport.arguments.fetch(:path), "alt=sse&key=secret"
    assert_equal "application/json", JSON.parse(transport.arguments.fetch(:body)).dig("generationConfig", "responseMimeType")
  end

  def test_vertex_uses_bearer_token_and_vertex_endpoint
    transport = Transport.new
    provider = LittleGhost::Providers::VertexAI.new(model: "gemini", project: "project", location: "us-central1",
      credential_resolver: ->(**) { "token" }, transport:)
    request = LittleGhost::ModelRequest.new(messages: [LittleGhost::Message.new(role: :user, content: "Hi")])

    provider.stream(request).to_a

    assert_equal "Bearer token", transport.arguments.dig(:headers, "authorization")
    assert_includes transport.arguments.fetch(:path), "projects/project/locations/us-central1"
    refute_includes transport.arguments.fetch(:path), "key="
  end

  class SequencedTransport
    attr_reader :requests

    def initialize(*responses)
      @responses = responses
      @requests = []
    end

    def stream(**arguments)
      @requests << arguments
      @responses.shift.each { |chunk| yield "data: #{JSON.generate(chunk)}\n\n" }
    end
  end

  def test_sends_the_original_function_name_back_with_a_tool_result
    call = {functionCall: {id: "tool-1", name: "lookup", args: {id: 1}}}
    transport = SequencedTransport.new(
      [{modelVersion: "gemini", candidates: [{content: {parts: [call]}, finishReason: "STOP"}]}],
      [{candidates: [{content: {parts: [{text: "done"}]}, finishReason: "STOP"}]}]
    )
    provider = LittleGhost::Providers::Gemini.new(api_key: "secret", model: "gemini", transport:)

    first_request = LittleGhost::ModelRequest.new(messages: [LittleGhost::Message.new(role: :user, content: "Hi")])
    response = provider.stream(first_request).to_a.last.data.fetch(:response)
    tool_use = response.message.content.grep(LittleGhost::Content::ToolUse).first
    result = LittleGhost::Content::ToolResult.new(tool_use_id: tool_use.id, content: "42", status: :success)
    follow_up = LittleGhost::ModelRequest.new(messages: [
      LittleGhost::Message.new(role: :user, content: "Hi"),
      LittleGhost::Message.new(role: :assistant, content: [tool_use]),
      LittleGhost::Message.new(role: :tool, content: [result])
    ])

    provider.stream(follow_up).to_a

    function_response = JSON.parse(transport.requests.last.fetch(:body)).fetch("contents")
      .flat_map { |message| message.fetch("parts") }.find { |part| part["functionResponse"] }
    assert_equal "lookup", function_response.dig("functionResponse", "name")
  end

  def test_falls_back_to_the_tool_use_id_for_a_result_the_provider_never_saw_a_call_for
    transport = Transport.new
    provider = LittleGhost::Providers::Gemini.new(api_key: "secret", model: "gemini", transport:)
    result = LittleGhost::Content::ToolResult.new(tool_use_id: "unseen-call", content: "42", status: :success)
    request = LittleGhost::ModelRequest.new(messages: [LittleGhost::Message.new(role: :tool, content: [result])])

    provider.stream(request).to_a

    function_response = JSON.parse(transport.arguments.fetch(:body)).fetch("contents")
      .flat_map { |message| message.fetch("parts") }.find { |part| part["functionResponse"] }
    assert_equal "unseen-call", function_response.dig("functionResponse", "name")
  end
end
