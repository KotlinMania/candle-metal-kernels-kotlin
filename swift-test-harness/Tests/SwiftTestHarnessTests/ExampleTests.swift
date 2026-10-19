#if canImport(Testing)
import Testing
import CandleMetalKernels

@Suite("CandleMetalKernels Swift Export Tests")
struct CandleMetalKernelsExportTests {
    @Test("Kernels select the requested library source through the Swift bridge")
    func testLibrarySource() {
        let kernels = Kernels()
        let device = metal.Device.Companion.shared.new(raw: nil)
        let library = kernels.loadLibrary(device: device, source: .Affine)
        #expect(library.sourceVal == AFFINE)
        #expect(!library.sourceVal.isEmpty)
        let error = MetalKernelError.InvalidInput(error: "missing dimensions")
        #expect(error.message == "Invalid input: missing dimensions")
    }
}
#elseif canImport(XCTest)
import XCTest
import CandleMetalKernels

final class CandleMetalKernelsExportTests: XCTestCase {
    func testLibrarySource() {
        let kernels = Kernels()
        let device = metal.Device.Companion.shared.new(raw: nil)
        let library = kernels.loadLibrary(device: device, source: .Affine)
        XCTAssertEqual(library.sourceVal, AFFINE)
        XCTAssertFalse(library.sourceVal.isEmpty)
        let error = MetalKernelError.InvalidInput(error: "missing dimensions")
        XCTAssertEqual(error.message, "Invalid input: missing dimensions")
    }
}
#endif
