"use client"

import { Button } from "@/components/ui/button"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Info, MessageCircle, Search, Brain, Save, Menu, X, ChevronDown, TrendingUp, Plane, ShoppingCart, Trash2 } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { useState } from "react"

export function Navbar() {
    const pathname = usePathname()
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

    const apiItems = [
        {
            href: "/save",
            label: "Add",
            icon: Save,
            isActive: pathname === "/save"
        },
        {
            href: "/search",
            label: "Search",
            icon: Search,
            isActive: pathname === "/search"
        },
        {
            href: "/ask",
            label: "Ask Question",
            icon: Brain,
            isActive: pathname === "/ask"
        },
        {
            href: "/delete",
            label: "Delete",
            icon: Trash2,
            isActive: pathname === "/delete"
        },
        {
            href: "/delete_all",
            label: "Delete all",
            icon: Trash2,
            isActive: pathname === "/delete_all"
        },
    ]

    const demoItems = [
        {
            href: "/chat-demo",
            label: "Memory Chat (LangGraph)",
            icon: MessageCircle,
            isActive: pathname === "/chat-demo"
        },
        {
            href: "/agent-demo",
            label: "Code Review Agent",
            icon: Brain,
            isActive: pathname === "/agent-demo"
        },
        {
            href: "/investment-demo",
            label: "Investment Agent",
            icon: TrendingUp,
            isActive: pathname === "/investment-demo"
        },
        {
            href: "/travel-demo",
            label: "Travel Agent",
            icon: Plane,
            isActive: pathname === "/travel-demo"
        },
        {
            href: "/retail-demo",
            label: "Retail Agent",
            icon: ShoppingCart,
            isActive: pathname === "/retail-demo"
        }
    ]

    const otherNavItems = [
        {
            href: "/api-docs",
            label: "Docs",
            icon: Info,
            isActive: pathname === "/api-docs"
        },
        {
            href: "/memory-info",
            label: "Settings",
            icon: Info,
            isActive: pathname === "/memory-info"
        }
    ]

    const isApiActive = apiItems.some(item => item.isActive)
    const isDemoActive = demoItems.some(item => item.isActive)

    const toggleMobileMenu = () => {
        setIsMobileMenuOpen(!isMobileMenuOpen)
    }

    const closeMobileMenu = () => {
        setIsMobileMenuOpen(false)
    }

    return (
        <nav className="bg-white/90 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-6">
                <div className="flex justify-between items-center h-16">
                    {/* Logo */}
                    <div className="flex items-center min-w-0">
                        <Link href="/" className="flex items-center space-x-2 sm:space-x-3">
                            <img
                                src="/Redis_logo.png"
                                alt="Redis Logo"
                                width={60}
                                height={60}
                                className="object-contain sm:w-[80px] sm:h-[80px]"
                            />
                            <div className="flex flex-col min-w-0 text-2xl">
                                <img
                                    src="/logo.png"
                                    alt="Logo"
                                    width={100}
                                    height={75}
                                    className="object-contain sm:w-[80px] sm:h-[80px]"
                                />

                            </div>
                        </Link>
                    </div>

                    {/* Desktop Navigation */}
                    <div className="hidden sm:flex items-center space-x-0">


                        {/* API Dropdown */}
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button
                                    variant={isApiActive ? "default" : "ghost"}
                                    className={cn(
                                        "flex items-center gap-2 bg-white",
                                        isApiActive && "text-red-600 hover:text-red-700 bg-white border-b-2 border-red-500 rounded-none hover:bg-white"
                                    )}
                                >
                                    API
                                    <ChevronDown className="w-4 h-4" />
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end" className="w-48">
                                {apiItems.map((item) => {
                                    const Icon = item.icon
                                    return (
                                        <DropdownMenuItem key={item.href} asChild>
                                            <Link href={item.href} className="flex items-center gap-2 w-full">
                                                <Icon className="w-4 h-4" />
                                                {item.label}
                                            </Link>
                                        </DropdownMenuItem>
                                    )
                                })}
                            </DropdownMenuContent>
                        </DropdownMenu>
                        {/* Demos Dropdown */}
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button
                                    variant={isDemoActive ? "default" : "ghost"}
                                    className={cn(
                                        "flex items-center gap-2 bg-white",
                                        isDemoActive && "text-red-600 hover:text-red-700 bg-white border-b-2 border-red-500 rounded-none hover:bg-white"
                                    )}
                                >
                                    Agent Demos
                                    <ChevronDown className="w-4 h-4" />
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end" className="w-48">
                                {demoItems.map((item) => {
                                    const Icon = item.icon
                                    return (
                                        <DropdownMenuItem key={item.href} asChild>
                                            <Link href={item.href} className="flex items-center gap-2 w-full">
                                                <Icon className="w-4 h-4" />
                                                {item.label}
                                            </Link>
                                        </DropdownMenuItem>
                                    )
                                })}
                            </DropdownMenuContent>
                        </DropdownMenu>

                        {/* Other nav items */}
                        {otherNavItems.map((item) => {
                            return (
                                <Link key={item.href} href={item.href}>
                                    <Button
                                        variant={item.isActive ? "default" : "ghost"}
                                        className={cn(
                                            "flex items-center gap-2 bg-white",
                                            item.isActive && "text-red-600 hover:text-red-700 bg-white border-b-2 border-red-500 rounded-none hover:bg-white"
                                        )}
                                    >
                                        {item.label}
                                    </Button>
                                </Link>
                            )
                        })}

                    </div>

                    {/* Mobile menu button */}
                    <div className="sm:hidden">
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={toggleMobileMenu}
                            className="p-2"
                            aria-label="Toggle mobile menu"
                        >
                            {isMobileMenuOpen ? (
                                <X className="w-5 h-5" />
                            ) : (
                                <Menu className="w-5 h-5" />
                            )}
                        </Button>
                    </div>
                </div>

                {/* Mobile Navigation Menu */}
                {isMobileMenuOpen && (
                    <div className="lg:hidden border-t border-gray-200 bg-white/95 backdrop-blur-sm">
                        <div className="px-2 pt-2 pb-3 space-y-1">
                            {/* Other nav items */}
                            {otherNavItems.map((item) => {
                                return (
                                    <Link key={item.href} href={item.href} onClick={closeMobileMenu}>
                                        <Button
                                            variant={item.isActive ? "default" : "ghost"}
                                            className={cn(
                                                "w-full justify-start gap-3 bg-white",
                                                item.isActive && "text-red-600 hover:text-red-700 bg-white border-l-4 border-red-500 rounded-none hover:bg-white"
                                            )}
                                        >
                                            {item.label}
                                        </Button>
                                    </Link>
                                )
                            })}

                            {/* Demos section header */}
                            <div className="px-3 py-2 text-sm font-medium text-gray-500 border-t border-gray-200 mt-2 pt-4">
                                Demos
                            </div>

                            {/* Demo items */}
                            {demoItems.map((item) => {
                                const Icon = item.icon
                                return (
                                    <Link key={item.href} href={item.href} onClick={closeMobileMenu}>
                                        <Button
                                            variant={item.isActive ? "default" : "ghost"}
                                            className={cn(
                                                "w-full justify-start gap-3 bg-white pl-6",
                                                item.isActive && "text-red-600 hover:text-red-700 bg-white border-l-4 border-red-500 rounded-none hover:bg-white"
                                            )}
                                        >
                                            <Icon className="w-4 h-4" />
                                            {item.label}
                                        </Button>
                                    </Link>
                                )
                            })}

                            {/* API section header */}
                            <div className="px-3 py-2 text-sm font-medium text-gray-500 border-t border-gray-200 mt-2 pt-4">
                                API
                            </div>

                            {/* API items */}
                            {apiItems.map((item) => {
                                const Icon = item.icon
                                return (
                                    <Link key={item.href} href={item.href} onClick={closeMobileMenu}>
                                        <Button
                                            variant={item.isActive ? "default" : "ghost"}
                                            className={cn(
                                                "w-full justify-start gap-3 bg-white pl-6",
                                                item.isActive && "text-red-600 hover:text-red-700 bg-white border-l-4 border-red-500 rounded-none hover:bg-white"
                                            )}
                                        >
                                            <Icon className="w-4 h-4" />
                                            {item.label}
                                        </Button>
                                    </Link>
                                )
                            })}
                        </div>
                    </div>
                )}
            </div>
        </nav>
    )
}
