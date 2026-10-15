#if canImport(Testing)
import Testing
import CandleMetalKernels

@Suite("CandleMetalKernels Swift Export Tests")
struct CandleMetalKernelsExportTests {
    @Test("CandleMetalKernels swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "CandleMetalKernels swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import CandleMetalKernels

final class CandleMetalKernelsExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "CandleMetalKernels swift module imported cleanly")
    }
}
#endif
