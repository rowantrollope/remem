import { HeroSection, MainContentSection, FeatureCardSection, CallToAction, Navbar } from "@/components"

export default function Home() {
    return (
        <div className="flex min-h-screen flex-col bg-[white] p-6">
            {/* Navigation */}
            <Navbar />

            {/* Hero Section */}
            <HeroSection />

            {/* Main Content */}
            <MainContentSection />

            {/* Features Section */}
            <FeatureCardSection />

            {/* CTA Section */}
            <CallToAction />
        </div>
    )
}
